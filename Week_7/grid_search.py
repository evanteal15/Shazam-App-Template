# https://github.com/dennisfarmer/F25-Shazam-GridSearch

import numpy as np
import pandas as pd
from itertools import product
import librosa

import os

from cm_visualizations import visualize_map_interactive

# updated add_noise that uses signal-to-noise ratio
#from cm_helper import create_samples, add_noise
from cm_helper import create_samples, preprocess_audio

from DBcontrol import init_db, connect, retrieve_song, create_tables, add_songs

from const_map import create_constellation_map
from hasher import create_hashes
from search import score_hashes

from parameters import set_parameters, read_parameters
from pathlib import Path

# microphone simulation
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, low_pass_filter, high_pass_filter

# score_hashes_no_index
from collections import defaultdict
from DBcontrol import retrieve_hashes

# measuring search time
import time

# GridViewer
import sqlite3
import pickle
import json

# logging errors in GridViewer
import traceback
import datetime

#
# 1 / 3: list samples to use to evaluate recognition performance
#
# https://marketplace.visualstudio.com/items?itemName=sukumo28.wav-preview

microphone_sample_list = [
    ("Plastic Beach (feat. Mick Jones and Paul Simonon)", "Gorillaz", "audio_samples/plastic_beach_microphone_recording.wav"),
    #("DOGTOOTH", "Tyler, The Creator", "tracks/audio/TylerTheCreator_DOGTOOTH_QdkbuXIJHfk.flac")
    # ...
]


microphone_sample_list = [
    {"title": v[0], "artist": v[1], "audio_path": v[2]} 
    for v in microphone_sample_list
    ]


def add_noise(audio, snr_dB: float):
    """
    add noise to librosa audio with a desired 
    signal-to-noise ratio (snr), measured in decibels.
    ```
    snr_dB = 10 log_10(P_signal / P_noise)

    P_signal / P_noise = (A_signal / A_noise)**2

    => snr_dB = 20 log_10(A_signal / A_noise)

    => A_signal / A_noise = 10**(snr_dB / 20)

    => A_noise = A_signal / 10**(SNR_dB / 20)
    ```
    
    We use the root mean square (RMS) of the amplitude for the 
    signal and noise. To achieve the desired SNR for our final 
    audio, we calculate a weight to multiply the noise by:

    ```
    A_noise_initial * noise_weight = A_noise

    => noise_weight = A_noise / A_noise_initial

    => noise_weight = rms_signal / (10**(snr_dB / 20)) / (rms_noise)

    (avoid division by zero by adding 1e-10 to denominator)
    ```
    """

    # brownian noise: x(n+1) = x(n) + w(n)
    #                 w(n) = N(0,1)

    def peak_normalize(x):
        # Normalize the audio to be within the range [-1, 1]
        return x / np.max(np.abs(x))

    static = np.random.normal(0, 1, audio.shape[0])
    static = np.cumsum(static)
    static = peak_normalize(static)
    y_cafe_ambience, sr_cafe = preprocess_audio("./audio_samples/cafe_ambience.flac")
    if len(y_cafe_ambience) < audio.shape[0]:
        y_cafe_ambience = np.concatenate(
            [y_cafe_ambience for i in range(int(np.ceil(len(y_cafe_ambience) / audio.shape[0])))]
        )
    if len(y_cafe_ambience) > audio.shape[0]:
        y_cafe_ambience = y_cafe_ambience[:audio.shape[0]]

            
    # 60/40 mix of conversation and static noise
    snr_noise_mix_dB = 10*np.log10(0.6 / 0.4)  # ~= 1.76 dB conversation to static ratio
    rms_cafe = np.sqrt(np.mean(y_cafe_ambience**2))
    rms_static = np.sqrt(np.mean(static**2))
    noise = (y_cafe_ambience + static*(rms_cafe / (10**( snr_noise_mix_dB / 20 )) / rms_static))

    rms_signal = np.sqrt(np.mean(audio**2))
    rms_noise = np.sqrt(np.mean(noise**2))

    noise_weight = rms_signal / (10**(snr_dB / 20)) / (rms_noise + 1e-10)

    audio_with_noise = (audio + noise*noise_weight)
    return peak_normalize(audio_with_noise)

def augment_samples(sr, snr):
    con = connect()
    cur = con.cursor()
    samples = []
    # TODO for members: Write this SQL query
    for sample in microphone_sample_list:
        cur.execute(
            "SELECT id AS song_id "
            "FROM songs "
            "WHERE title LIKE ? "
            "AND artist = ?",
            (sample["title"], sample["artist"])
        )

        res = cur.fetchone()
        if res: 
            song_id = res[0]
        else:
            raise ValueError(f'\'{sample["title"]}\' by \'{sample["artist"]}\' not found.\nSpelling of name/artist might be different from the spelling in tracks/audio/tracks.csv?')
        
        sample_slices = create_samples(sample["audio_path"], sr, n_samples = 20)
        sample_slices_noisy = [add_noise(audio, snr) for audio in sample_slices]
        samples.extend([
            {"song_id": song_id,
             "microphone": s[0],
             #"mic_noisy": s[1]
             } for s in zip(sample_slices, sample_slices_noisy)
        ])
    return samples

def simulate_microphone_from_source(y, sr, drc_threshold=-40, drc_ratio=4.0, drc_attack=10, drc_release=50):
    """
    does two things:

    1) Band-pass filtering (keep between 120 Hz and 3500 Hz, 
    with bandwidth extending up to 4000 Hz)

    2) Dynamic Range Compression using pydub
    """
    ##################################
    # Harmonic-Percussive
    # Source Separation (HPSS)
    # maintain percussion voices
    # after filtering and dynamic
    # range compression
    ##################################
    #D = librosa.stft(y)
    #H, P = librosa.decompose.hpss(D)
    #y_harmonic = librosa.istft(H)
    ## should always pad with 256 zeros  (128 [...] 128)
    #y_harmonic = np.pad(y_harmonic,
                          #int(max(len(y) - len(y_harmonic), 0)/2))
    #y_percussive = librosa.istft(P)
    #y_percussive = np.pad(y_percussive,
                          #int(max(len(y) - len(y_percussive), 0)/2))



    ##################################
    # bandwidth limitation 
    # via resampling
    ##################################
    target_sr = 8000
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    y = librosa.resample(y, orig_sr=target_sr, target_sr=sr)

    ##################################
    # convert to 16 bit integer
    # (librosa to pydub)
    ##################################

    max_amplitude = np.iinfo(np.int16()).max
    y_int16 = (y * max_amplitude).astype(np.int16)

    y_audioseg = AudioSegment(
        data = y_int16.tobytes(),
        frame_rate = sr,
        sample_width = y_int16.dtype.itemsize,
        channels=1)
    
    ##################################
    # low pass filter: 3500 Hz cutoff
    # higher frequencies reduced by 
    # 6 dB per octave above cutoff
    ##################################
    # https://github.com/jiaaro/pydub/blob/master/pydub/effects.py#L222
    y_audioseg = low_pass_filter(y_audioseg, 3500)

    ##################################
    # high pass filter: 120 Hz cutoff
    # lower frequencies reduced by 
    # 6 dB per octave below cutoff
    ##################################
    y_audioseg = high_pass_filter(y_audioseg, 120)

    ##################################
    # dynamic range compression
    ##################################
    # https://github.com/jiaaro/pydub/blob/master/pydub/effects.py#L116
    # https://en.wikipedia.org/wiki/Dynamic_range_compression
    y_audioseg = compress_dynamic_range(
        y_audioseg,
        threshold=drc_threshold,
        ratio = drc_ratio,
        attack = drc_attack,
        release=drc_release
    )

    ##################################
    # convert back to 32 bit float
    # (pydub to librosa)
    ##################################
    y_compressed = (np.array(y_audioseg.get_array_of_samples()) / max_amplitude).astype(np.float32)
    #return np.clip(y_compressed + (y_percussive * 0.3), -1, 1)
    return np.clip(y_compressed, -1, 1)


def sample_from_source_audios(n_songs: int = None, tracks_dir = "./tracks", sr=11025, snr=None) -> list[dict]:
    """
    returns samples with song_id labels: `[{"song_id": .., "microphone": [..,..,..]}, ...]`
    This function attempts to simulate the act of recording many microphone samples
    by doing some dynamic range compression and filtering of the source audio files

    spectrogram looks similar to actual microphone recordings, primary trait 
    of microphone recordings compared to the source audio is having a lower 
    amplitude in the waveform on average (lower dynamic range)

    computes the same samples each time since random sampling uses `seed=1`, => reproducable samples

    ## Arguments:
    - `n_songs`: optionally take the top n songs from the top of the `tracks_dir` dataset.
        Default is to use all songs
    - `snr`: optionally specify a signal-to-noise ratio for adding noise. 
        Default is to not add noise. 
        Can be done afterwards to save computation on recreating samples: 
        `sample_slices = [add_noise(audio, snr) for audio in sample_slices]`

    assumes that song_ids correspond to order that tracks appear in csv

    (first row => song_id = 1, ...)
    """
    samples = []
    tracks_dir = Path(tracks_dir)
    df = pd.read_csv(tracks_dir/"tracks.csv")
    for idx, track in df.iterrows():
        song_id = idx + 1
        if n_songs is not None:
            if song_id > n_songs:
                break
        sample_slices = create_samples(tracks_dir/track["audio_path"], sr, n_samples=5, n_seconds=5, seed=1)
        sample_slices = [simulate_microphone_from_source(s, sr) for s in sample_slices]
        if snr is not None:
            sample_slices = [add_noise(audio, snr) for audio in sample_slices]
        samples.extend([
            {"song_id": song_id,
             "microphone": s
            } for s in sample_slices
        ])
    return samples


#
# 2 / 3: Compute some metrics using output of search.py
#

def std_of_deltaT(song_id, time_pair_bins, sr=11025):
    """
    heuristic that measures the distribution of deltaT values

    results in standard deviation between 0 ms and 1000 ms

    lower is better - suggests that the time offsets are behaving consistently
    """
    time_pair_bin = time_pair_bins[song_id]

    # 1) take the difference in time between sourceT and sampleT
    #    values represent the offset suggested by the presence of the 
    # respective hash match
    # ex: offset = 40
    # data might look like [-4, 5, 8, 40, 40, 40, ..., 40, 40, 40, 56, 68, 80]
    deltaT_vals = sorted([sourceT-sampleT for (sourceT, sampleT) in time_pair_bin])

    #    Potential idea: then, take the difference between adjacent deltaT values
    #    removes influence of offset / outlier offset values
    #    values are close to zero for consistent time offsets
    # ex: [9, 3, 0, 0, 0, ..., 0, 0, 16, 12, 12]
    #second_order_differences = [deltaT_vals[i] - deltaT_vals[i-1] 
                                #for i in range(1, len(deltaT_vals))]
    # not doing this step makes the metric more interpretable
                                

    # 3) compute the standard deviation of these differences, measured in seconds
    #    return a metric between 0 and 1000
    #    (stddev between 0 and 1000 milliseconds, 0 and 1 seconds)
    #
    # T is in units of STFT bins, 
    # multiply by 1000*(hop_length / sr) to get in units of milliseconds
    #
    # Unit conversion:
    # bins * ((n_samples_to_jump/bin) / (samples/second)) = bins * (seconds / bin) = seconds
    # seconds * (1000 ms / second) = milliseconds
    # std(x * v) = x * std(v)
    hop_length = 1024 // 2  # sidenote: not 1024 + (1024 // 2) as in cm_helper.py

    return min(np.std(deltaT_vals) * (hop_length/sr), 1000)


def count_hash_matches(song_id, time_pair_bins, n_sample_hashes):
    """
    naive heuristic that counts the number of hash matches between sample and source
    
    note: there can be repeated matches for a single hash value present in sample

    higher is better - suggests that many hashes show up in both sample and source
    """
    n_matches = len(time_pair_bins[song_id]) 
    return n_matches


def compute_performance_metrics(song_id, time_pair_bins, n_sample_hashes, sr=11025):
    """
    given the output from search.py:score_hashes(), computes metrics that
    evaluate how close a given song_id matches the sample audio

    returns a dictionary containing metrics

    ## Metrics:
     
    - `std_of_deltaT`: less is better, (0-1000 ms)
    - `n_hash_matches`: more is better
    - `prop_hash_matches`: `min(cout_hash_matches / n_sample_hashes, 1)
    
    """
    metrics = {
        "std_of_deltaT": std_of_deltaT(song_id, time_pair_bins, sr),
        "n_hash_matches": count_hash_matches(song_id, time_pair_bins, n_sample_hashes),
        "n_sample_hashes": n_sample_hashes
    }
    metrics["prop_hash_matches"] = min(metrics["n_hash_matches"] / metrics["n_sample_hashes"], 1)
    return metrics

###########################################

def recompute_hashes():
    with connect() as con:
        con.execute("DELETE FROM hashes")
        con.commit()
    from DBcontrol import compute_source_hashes
    compute_source_hashes()

class GridViewer():
    """
    save summary statistics for each combination of parameters used in grid search

    use for visualization of the four main audio fingerprint system parameters:
    - *Reliability*: can the model actually recognize tracks?
    - *Robustness*: how resistant is the model to noise?
    - *Fingerprint size*: how much disk space is used?
    - *Search speed*: how long does it take to search for a match?
    """
    def __init__(self, parameter_grid):
        self.database = sqlite3.connect(":memory:")
        self.filename = "sql/gridviewer.db"
        self.cursor = self.database.cursor()
        self.cursor.execute(
            "CREATE TABLE paramsets ( "
            "paramset_idx INTEGER PRIMARY KEY, "
            "cm_window_size INTEGER, "
            "candidates_per_band INTEGER, "
            "bands BLOB, "
            "fanout_t INTEGER, "
            "fanout_f INTEGER);"
        )
        for paramset_idx, (cm_window_size, candidates_per_band, bands, fanout_t, fanout_f) in enumerate(parameter_grid):
            self.cursor.execute(
                "INSERT INTO paramsets "
                "(paramset_idx, cm_window_size, candidates_per_band, bands, fanout_t, fanout_f) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (paramset_idx, cm_window_size, candidates_per_band, pickle.dumps(bands), fanout_t, fanout_f)
            )
        self.database.commit()

        self.cursor.execute(
            "CREATE TABLE results ( "
            "paramset_idx INTEGER, "
            "snr FLOAT, "
            "proportion_correct FLOAT, "
            "avg_search_time_s FLOAT, "
            "PRIMARY KEY (paramset_idx, snr) "
            ")"
        )

        self.cursor.execute(
            "CREATE TABLE fingerprint_densities ( "
            "paramset_idx INTEGER, "
            "song_title TEXT, "
            "song_id INTEGER, "
            "n_hashes INTEGER, "
            "n_seconds FLOAT, "
            "hashes_per_second FLOAT, "
            "fingerprint_size_MB FLOAT, "
            "source_audio_size_MB FLOAT, "
            "compression_ratio FLOAT, "
            "compression_ratio_inv FLOAT, "
            "PRIMARY KEY (paramset_idx, song_id)"
            ")"
        )

        self.cursor.execute(
            "CREATE TABLE database_size ( "
            "paramset_idx INTEGER PRIMARY KEY, "
            "total_fingerprint_size_MB FLOAT, "
            "hashes_MB FLOAT, "
            "hash_index_MB FLOAT "
            ")"
        )

        self.cursor.execute(
            "CREATE TABLE errors ("
            "paramset_idx INTEGER, "
            "time_of_error TEXT, "
            "stacktrace TEXT, "
            "snr FLOAT, "
            "PRIMARY KEY (paramset_idx, snr)"
            ")"
        )

    def add_result(self, paramset_idx, results, snr):
        """
        `param_idx` is the index of `parameter_grid=list(product(...))` corresponding to parameters
        used to achieve `results`.
        """
        summary = self.compute_results_summary(results)
        self.cursor.execute(
            "INSERT INTO results "
            "(paramset_idx, snr, proportion_correct, avg_search_time_s) "
            "VALUES (?, ?, ?, ?)",
            (paramset_idx, snr, summary["proportion_correct"], summary["avg_search_time_s"])
        )
        self.database.commit()
    
    def add_fingerprint_size(self, paramset_idx, database_size: tuple[float,float], fingerprint_densities_df: pd.DataFrame):
        self.cursor.execute(
            "INSERT INTO database_size "
            "(paramset_idx, total_fingerprint_size_MB, hashes_MB, hash_index_MB) "
            "VALUES (?, ?, ?, ?)",
            (paramset_idx, database_size[0] + database_size[1], database_size[0], database_size[1])
        )
        for i, row in fingerprint_densities_df.iterrows():
            self.cursor.execute(
                "INSERT INTO fingerprint_densities "
                "(paramset_idx, song_title, song_id, n_hashes, n_seconds, hashes_per_second, "
                "fingerprint_size_MB, source_audio_size_MB, compression_ratio, compression_ratio_inv) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (paramset_idx, row["song_title"], row["song_id"], row["n_hashes"], row["n_seconds"],
                 row["hashes_per_second"], row["fingerprint_size_MB"], row["source_audio_size_MB"],
                 row["compression_ratio"], row["compression_ratio_inv"])
            )
        self.database.commit()
    
    def compute_results_summary(self, results):
        """
        summarize the results table to be stored with 
        its corresponding parameters and snr
        """
        results_df = pd.DataFrame(results)
        return {
            "proportion_correct": results_df["correct"].mean(),
            "avg_search_time_s": results_df["search_time_with_index"].mean(),
            "misc_info": {
                "avg_search_time_without_index": results_df["search_time_without_index"].mean(),
            }
        }

    def paramset_idx_to_params(self, paramset_idx: int):
        res = self.cursor.execute("SELECT * FROM paramsets WHERE paramset_idx = ?", (paramset_idx,)).fetchone()
        return {
            "cm_window_size": res[1], 
            "candidates_per_band": res[2], 
            "bands": pickle.loads(res[3]), 
            "fanout_t": res[4], 
            "fanout_f": res[5]
            }
    
    def to_sqlite(self, filename="gridviewer.db"):
        # .iterdump()
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError as e:
                print("os error")
                print(e)
                pass
        self.filename = filename
        with sqlite3.connect(filename) as export_db:
            self.database.backup(export_db)


    def log_exception(self, paramset_idx, stacktrace, snr):
        """
        obtain stacktrace string with `traceback.format_exc()` inside `except` clause.
        """
        time_of_error = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute(
            "INSERT INTO errors "
            "(paramset_idx, time_of_error, stacktrace, snr) "
            "VALUES (?, ?, ?, ?)",
            (paramset_idx, time_of_error, stacktrace, snr)
        )
        self.database.commit()



def score_hashes_no_index(hashes: dict[int, tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, set[int, int]]]:
    """
    same as `search.py:score_hashes()`, except does not use the `hashes` table index when retrieving matching hashes

    used for computing search time metrics to show effectiveness of database index
    """
    con = connect()
    cur = con.cursor()
    time_pair_bins = defaultdict(set)
    for address, (sampleT, _) in hashes.items():
        # DBcontrol.py:retrieve_hashes(), without using index
        cur.execute("SELECT hash_val, time_stamp, song_id FROM hashes NOT INDEXED WHERE hash_val = ?", (address,))
        matching_hashes = cur.fetchall()
        if matching_hashes is not None:
            for _, sourceT, song_id in matching_hashes:
                time_pair_bins[song_id].add((sourceT, sampleT))
    scores = {}
    for song_id, time_pair_bin in time_pair_bins.items():
        deltaT_values = [sourceT - sampleT for (sourceT, sampleT) in time_pair_bin]
        hist, bin_edges = np.histogram(deltaT_values, bins=max(len(np.unique(deltaT_values)), 10))
        scores[song_id] = hist.max()
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    con.close()
    return scores, time_pair_bins


def perform_recognition_test(n_songs=None, samples=None, perform_no_index_searches=False, store_microphone_samples=False):
    """
    returns a tuple:
    `(n_correct / n_samples, performance results for each sample)`

    first value is the proportion of correct recognitions

    samples can be:
    - specified based on slicing of samples in `microphone_sample_list` using `grid_search.py:augment_samples()`
    - created using `grid_search.py:sample_from_source_audios()`, which simulates the dynamic range compression common in microphone recordings

    ## Arguments
    - `perform_no_index_searches`: track metrics for retrieving hashes without using the database's index. Takes a long time, only useful for giving an intuition of the trade off between index storage size and search speed.
    - `store_microphone_samples`: include in the performance results the actual samples used for recognition. Useful for perceptually evaluating how the altered samples sound, but takes a lot of storage. Can be recomputed from the sample generating method anyways.
    """
    #init_db()
    #init_db(n_songs=n_songs)
    #if samples is None:
        #snr = ???
        #samples = augment_samples(sr=sr, snr=snr)

    sr = 11025
    results = []
    for sample in samples:
        result = {}
        ground_truth_song_id = sample["song_id"]

        # peaks -> hashes
        constellation_map = create_constellation_map(sample["microphone"], sr=sr)
        hashes = create_hashes(constellation_map, None, sr)

        # hashes -> metrics
        start_time = time.time()
        scores, time_pair_bins = score_hashes(hashes)
        search_time_with_index = time.time() - start_time

        if perform_no_index_searches:
            start_time = time.time()
            score_hashes_no_index(hashes)
            search_time_without_index = time.time() - start_time
        else:
            search_time_without_index = None

        n_sample_hashes = len(hashes)
        n_potential_matches = min(len(scores), 5)
        metrics_per_potential_match = {}
        for potential_song_id, potential_score in scores[:n_potential_matches]:
            metrics_per_potential_match[potential_song_id] = compute_performance_metrics(
                potential_song_id, time_pair_bins, n_sample_hashes, sr
            )
            metrics_per_potential_match[potential_song_id]["histogram_max_height"] = potential_score
        

        # store metrics for each sample
        result = {
            "ground_truth": ground_truth_song_id,
            "prediction": scores[0][0],
            "correct": ground_truth_song_id == scores[0][0],
            "metrics": metrics_per_potential_match,
            "search_time_with_index": search_time_with_index,
            "search_time_without_index": search_time_without_index,
            "microphone_audio": None,
        }
        # optionally keep the sample used 
        # (for perceptual evaluation, 
        # takes up lots of storage, 
        # can easily be recomputed later 
        # since sample generation is 
        # seeded and you can pass in desired SNR)
        if store_microphone_samples:
            result["microphone_audio"] = sample["microphone"]

        results.append(result)

    return sum(r["correct"] for r in results) / len(samples), results

def compute_database_size():
    """
    returns `(all_hashes_size_MB, hash_index_size_MB)`

    https://sqlite.org/dbstat.html
    """
    with connect() as con:
        cur = con.cursor()
        cur.execute(
            "SELECT name, SUM(pgsize) / (1024.0*1024.0) as size_in_mb "
            "FROM dbstat "
            "GROUP BY name "
            "HAVING name IN ('hashes', 'idx_hash_val') "
            "ORDER BY "
                "CASE name "
                "WHEN 'hashes' THEN 1 "
                "ELSE 2 "
                "END"
        )
        res = cur.fetchall()
        size_hashes = res[0][1]
        if len(res) == 1 or res[1][0] != "idx_hash_val":
            # idx_hash_val doesn't exist
            size_hash_index = 0
        else:
            size_hash_index = res[1][1]

        return size_hashes, size_hash_index


#def compute_total_length_of_source_audio():
    #"""
    #number of seconds of source audio in database

    #can be used to normalize fingerprint size to Megabytes per second of audio
    #"""
    #with connect() as con:
        #cur = con.cursor()
        #cur.execute(
            #"SELECT SUM(duration_s) as total_duration_s "
            #"FROM songs"
        #)
        #total_duration_s = cur.fetchone()[0]
    
    #return total_duration_s

def compute_n_hashes_per_songid():
    with connect() as con:
        cur = con.cursor()
        cur.execute(
            """
            SELECT 
                song_id, 
                count(*) as n_hashes, 
                songs.duration_s, 
                count(*) / songs.duration_s as hashes_per_second
            FROM hashes 
            JOIN songs ON songs.id = hashes.song_id 
            GROUP BY hashes.song_id
            """
        )
        res = cur.fetchall()
    
    return pd.DataFrame([{
        "song_id": r[0], 
        "n_hashes": r[1],
        "n_seconds": r[2],
        "hashes_per_second": r[3]}
        for r in res])

def compute_fingerprint_density():
    """
    given the current parameters set in `parameters.json`, return 
    information about the size of the fingerprints in the database
    """
    fingerprint_density_info = compute_n_hashes_per_songid()
    db_size_MB = compute_database_size()
    total_db_size_MB = db_size_MB[0] + db_size_MB[1]
    n_hashes = fingerprint_density_info["n_hashes"].sum()
    fingerprint_density_info["fingerprint_size_MB"] = fingerprint_density_info[
        "n_hashes"
        ].apply(lambda n: (n*total_db_size_MB)/n_hashes)
    fingerprint_density_info["source_audio_size_MB"] = fingerprint_density_info[
        "song_id"
        ].apply(lambda song_id: os.stat(retrieve_song(song_id)["audio_path"]).st_size / (1024.0*1024.0))
    fingerprint_density_info["compression_ratio"] = fingerprint_density_info["fingerprint_size_MB"] / fingerprint_density_info["source_audio_size_MB"] 
    fingerprint_density_info["compression_ratio_inv"] = fingerprint_density_info["compression_ratio"].apply(lambda r: r ** -1)
    fingerprint_density_info["song_title"] = fingerprint_density_info[
        "song_id"].apply(lambda song_id: retrieve_song(song_id)["title"])

    title = fingerprint_density_info.pop("song_title")
    fingerprint_density_info.insert(0, "song_title", title)
    return fingerprint_density_info



def run_grid_search(parameter_space_subset: dict[str,list] = None, signal_to_noise_ratios: list[int|float] = None, n_songs=None):
    """
    ## Arguments

    - `parameters_space_subset`: a dictionary with keys being parameter names 
    from `parameters.json`, and values being a list of values for the 
    parameter to search over in the grid search. 
        - Default is some sensible defaults, testing `candidates_per_band =6 and =2`.
    - `signal_to_noise_ratios`: a list of numbers specifying the different amounts 
    of noise to apply to the test samples to measure noise robustness. 
    Measured in decibels. SNR=0 represents 50/50 mix of audio and noise, 
    negative numbers indicate more noise and positive numbers indicate more signal 
    (music audio). 
        - Default is `[3,6]`.
    - `n_songs`: optional parameter to use the top `n_songs` from the tracks library
    instead of generating samples from and testing over all tracks. Using a subset of the top
    few tracks helps with getting a general idea of how the parameters affect 
    fingerprint system parameters without having to wait a long time.
        - Default is to use all tracks in library.

    ## Example:
    ```
    parameter_space_subset = {
            "cm_window_size": [10, 20],
            "candidates_per_band": [5, 6, 7, 8]
    }
    run_grid_search(parameter_space_subset)

    # runs the following evaluations:
    # (cm_window_size, candidates_per_band):
    # [(10, 5), (10, 6), (10, 7), (10, 8), (20, 5), (20, 6), (20, 7), (20, 8)]
    # unspecified parameters are set to sensible defaults.
    ```

    """
    results = {}

    paramspace = {
        "cm_window_size": [10],
        "candidates_per_band": [6, 2],
        "bands": [[(0,10),(10,20),(20,40),(40,80),(80,160),(160,512)]],
        "fanout_t": [100],
        "fanout_f": [1500]
    }

    if parameter_space_subset is not None:
        for k, v in parameter_space_subset.items():
            paramspace[k] = v

    if signal_to_noise_ratios is None:
        signal_to_noise_ratio_grid = [3, 6]
    else: 
        signal_to_noise_ratio_grid = signal_to_noise_ratios

    grid_cm_window_size = paramspace["cm_window_size"]
    grid_candidates_per_band = paramspace["candidates_per_band"]
    grid_bands = paramspace["bands"]
    grid_fanout_t = paramspace["fanout_t"]
    grid_fanout_f = paramspace["fanout_f"]

    parameter_grid = list(
        product(
            grid_cm_window_size,
            grid_candidates_per_band,
            grid_bands,
            grid_fanout_t,
            grid_fanout_f
            )
        )

    grid_viewer = GridViewer(parameter_grid)

    set_parameters()
    create_tables()
    add_songs("./tracks", n_songs)

    samples = sample_from_source_audios(n_songs=n_songs)

    for paramset_idx, (cm_window_size, candidates_per_band, bands, fanout_t, fanout_f) in enumerate(parameter_grid):

        set_parameters(
            cm_window_size=cm_window_size,
            candidates_per_band=candidates_per_band,
            bands=bands,
            fanout_t=fanout_t,
            fanout_f=fanout_f
        )

        all_parameters = read_parameters("all_parameters")

        print(f"paramset_idx {paramset_idx}: {paramset_idx+1} / {len(parameter_grid)}")
        print(all_parameters)

        try:
            recompute_hashes()
        except:
            print("\t exception occured")
            stacktrace=traceback.format_exc()
            grid_viewer.log_exception(
                paramset_idx=paramset_idx, 
                stacktrace=stacktrace,
                snr=snr
                )
            continue

        # record size of database and size of individual fingerprints
        # for each set of parameters 
        database_size = compute_database_size()
        fingerprint_densities_df = compute_fingerprint_density()
        grid_viewer.add_fingerprint_size(paramset_idx, database_size, fingerprint_densities_df)

        for j, snr in enumerate(signal_to_noise_ratio_grid):
            print(f"\tsnr={snr}: {j+1} / {len(signal_to_noise_ratio_grid)}")
            samples_with_noise = [{"song_id": audio["song_id"], "microphone": add_noise(audio["microphone"], snr)} for audio in samples]
            try:
                proportion_correct, results = perform_recognition_test(n_songs, samples_with_noise)
                grid_viewer.add_result(paramset_idx, results, snr)
                #if proportion_correct > max_proportion_correct:
                    #max_proportion_correct = proportion_correct
                    #best_params = all_parameters
                    #best_results = results
            except:
                # there can be invalid combinations of parameters
                #
                # example: first band is 5 freq bins tall, cm_window_size is 5, and
                # we attempt to extract candidates_per_band=40 peaks from
                # the bottom left spectrogram window
                #
                # if we run into an error, log exception traceback
                print("\t exception occured")
                stacktrace=traceback.format_exc()
                grid_viewer.log_exception(
                    paramset_idx=paramset_idx, 
                    stacktrace=stacktrace,
                    snr=snr
                    )


    grid_viewer.to_sqlite(filename="sql/gridviewer.db")
    return grid_viewer
    


    
#def main():
    #pb_source = "tracks/audio/Gorillaz_PlasticBeachfeatMickJonesandPaulSimonon_pWIx2mz0cDg.flac"
    #pass
    #from cm_helper import preprocess_audio
    #y, sr = preprocess_audio(pb_source)
    #y = np.clip(y, -1, 1)
    #y_compressed = simulate_microphone_from_source(y,sr)
    #import soundfile as sf
    #sf.write("pb_compressed.wav", y_compressed, sr)

#if __name__ == "__main__":
    #main()






