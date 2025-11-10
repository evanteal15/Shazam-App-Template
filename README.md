# Shazam-App-Template

This will be the backbone of your Shazam clone applications for the rest of the semester. You may fork or copy files from this repository into your own repository in order to get started.

MDST Shazam Library - [download `tracks/` dataset as zip file here](https://drive.google.com/drive/folders/1Ui7o23sJjZB6tYUnoAffurmK0YB5nRVv?usp=sharing)
    - extract `week3tracks.zip` to `tracks/` directory inside of `Shazam-App-Template`
    - optional: load `gridviewer.db` to visualize performance of recognition algorithm using different parameter sets (see Week 7)

## How to start an Expo App

Create an expo account to start

Run the following commands in your terminal:
```bash
npx expo login -u YOUR_USERNAME -p YOUR_PASSWORD
npx create-expo-app@latest
npx expo start --tunnel 
```

Now that you've done this, copy the `Recorder.tsx` from the `Week_6/` directory to your `/app/(tabs)/` directory and begin working on the TODOs!


Provided helper files (no need to implement any code):

- `dataloader.py` - interface to work with a tracks dataset as a list of dictionaries (`dataloader.load()`)
- `cm_helper.py` - audio preprocessing, STFT computation, test sample creation
- `cm_visualizations.py` - plots spectrograms with peaks
- `DBcontrol.py` - database for managing many hashes
- `predict_song.py` - creates a `/predict` endpoint for interfacing with music recognition model

Files:

- `const_map.py` - constellation mapping
- `hasher.py` - creates hashes for representing pairs of peaks in database
- `search.py` - detailed implementation of audio search

Testing:

- `test_hash.py` - creates audio fingerprints
- `test_search.py` - sends a request to `/predict` endpoint

# Weeks 1-4

- [Project Intro](https://docs.google.com/presentation/d/1zfACjefKNI2SxUwyjICdXe_Cc1dKNuPlsfJnkOWKs7I/edit?usp=sharing)
- [Week 1 Slides - Introduction](https://docs.google.com/presentation/d/1tnqeYWHYlpvawnyNVfE3f95q3EHNDDlER3AwjsX0hdI/edit?usp=sharing)
- [Week 1 Repo](https://github.com/evanteal15/f25-shazam-clone-w1/tree/main)
- [Week 2 Slides - Fourier Transforms and Spectrograms](https://docs.google.com/presentation/d/1gzf0cIOUEAgEXo3vhXCraE0z4VufTNPwGH04obc6YVs/edit?usp=sharing)
- [Week 2 Repo](https://github.com/evanteal15/f25-shazam-clone-w2)
- [Week 3 Slides - Constellation Mapping, Combinatorial Hashing](https://docs.google.com/presentation/d/19fj9Kg58Cis_cMwXxcogEZqUZJeLFWYO1qcOLnea8xo/edit?usp=sharing)
- [Week 4 Slides - Audio Search using Audio Fingerprints](https://docs.google.com/presentation/d/1qd3xymwVmRnYa82Aees02pbQgLPBiSX_ctFwtJgPWrw/edit?usp=sharing)

# Week 5

- [Week 5 Slides - SQL, Evaluation (Buffer Week)](https://docs.google.com/presentation/d/1oNsmJnGLWtdSM7dvNCPTMTKxurSGHeCTk-tC6Q5EJIg/edit?usp=sharing)
- `test_add_song.py` - Send a request to `/add_song` endpoint
- `predict_song.py` -  Create a `/add_song` endpoint
- `DB_adder.py` - TODO: Write SQL queries that add an audio fingerprint to the database

# Week 6

- [Week 6 Slides - Expo](https://docs.google.com/presentation/d/1EzMEb6Ys__cdj71dZoUrnIjd7Qhoe4QodGnrk9UChWY/edit?usp=sharing)
- `code_for_predict_song.txt` - converts audio recieved via POST request to flac files in `predict_song.py`
- `Recorder.tsx` - TODO: React Native frontend application for sending requests to `predict_song.py` backend.
    - sends POST requests to `/predict`, `/add_song`

# Week 7

- [Week 7 Slides - App Customization](https://docs.google.com/presentation/d/1ntDU_eE7OyXqVKCb_PSP5gKcHzf4S4O_JvaPvpU6tZo/edit?usp=sharing)
- `grid_search.py` - performs a parameter sweep and stores metrics for each combination of parameters in `gridviewer.db`
- `grid_search.ipynb` - used for data analysis of `gridviewer.db`
- `parameters.py`: globally set parameters in a json file for access by algorithm
    - [F25-Shazam-GridSearch - export results of parameter sweep to a SQLite database for analysis](https://github.com/dennisfarmer/F25-Shazam-GridSearch/tree/master)
    - [musicdl - create custom music dataset using free Spotify API credentials](https://github.com/dennisfarmer/musicdl)
    - [precomputed `gridviewer.db` on Google Drive](https://drive.google.com/file/d/1cv1LpDl98PW5T54Sdc4CSBDxhQOPWwKZ/view?usp=drive_link) that can be loaded and queried

# Week 8

...

---

# Some helpful code snippets

## Loading audio dataset from [`week3tracks.zip`](https://drive.google.com/file/d/11zWYTmj4jxXbsz6bnfiSyVFxZQA4ZVma/view?usp=drive_link) to `tracks/`:

1. download [`week3tracks.zip`](https://drive.google.com/file/d/11zWYTmj4jxXbsz6bnfiSyVFxZQA4ZVma/view?usp=drive_link) from Google Drive
2. extract zip to `tracks/` folder inside of your `Shazam-App-Template` repository

The below code does this using the provided functions inside of `dataloader.py`

```python
from dataloader import extract_zip, load

# download tracks from Google Drive link above:
# week3tracks_tiny.zip       mp3 format, 92 MB
# week3tracks.zip           flac format, 1 GB

# extract zip archive (can do this from your file manager also)
extract_zip(zip_file="./week3tracks_tiny.zip", audio_directory = "./tracks")

# load track information as a list of dictionaries
tracks_info = load(audio_directory = "./tracks")

print(tracks_info[0])
#{
    #'youtube_url': 'https://www.youtube.com/watch?v=pWIx2mz0cDg',
    #'title': 'Plastic Beach (feat. Mick Jones and Paul Simonon)',
    #'artist': 'Gorillaz',
    #'artwork_url': 'https://i.scdn.co/image/ab67616d0000b273661d019f34569f79eae9e985',
    #'audio_path': 'tracks/audio/Gorillaz_PlasticBeachfeatMickJonesandPaulSimonon_pWIx2mz0cDg.mp3'
#}
```

## `predict_song.py`: separate `init_db()` from `app.run()`:

Flask auto refreshes `predict_song.py` whenever a file change is detected, which is helpful for debugging but takes a while if you have a large dataset of tracks. 

It might be useful to create something like an `initialize_database.py` that looks like the following:

```py
from DBcontrol import init_db
init_db(n_songs = None)  # load all songs
print("database initialized")
```

Then, just have the following at the bottom of your `predict_song.py`:
```py
if __name__ == '__main__':
    # Initialize the database using our command for now
    # init_db(n_songs=4)  # <-- comment out this line

    # Run the Flask app at this given host and port
    app.run(host='0.0.0.0', port=5003, debug=True)
```

Optionally, you could also organize your startup commands using a [Makefile](https://eecs280staff.github.io/tutorials/make.html):
```sh
# contents of Shazam-App-Template/Makefile
database:
    # replace `env/bin/python` with your environment (output of `which python`)
	env/bin/python ./initialize_database.py

backend:
    # replace `env/bin/python` with your environment (output of `which python`)
	env/bin/python ./predict_song.py

frontend:
    # replace `shazam` with path to your Expo app
	cd shazam && npx expo start --tunnel
```

Then run the following:
```sh
make database
# (in seperate terminals):
make backend
make frontend
```

## Loading precomputed grid search results:

Running grid search can take a really long time, so we've precomputed a grid search results dataset that you can load using the provided functions in `grid_search.py`.

```py
from grid_search import GridViewer
from parameters import set_parameters

###############################################################
# TODO: copy these lines to load precomputed
###############################################################

grid_viewer = GridViewer()
grid_viewer.from_sqlite("gridviewer.db")  # from google drive link

###############################################################
# initialize fingerprint database using 
# selected parameter set
#
# example: use paramset corresponding to paramset_idx = 0
paramset = grid_viewer.paramset_idx_to_params(paramset_idx = 0)

# **dict notation:
#         convert {"k1": "v1", "k2: "v2"}
#               -> k1="v1", k2="v2"
set_parameters(**paramset)

# compute fingerprints for all songs in tracks/ folder
init_db(n_songs=None)

# optional TODO: analyze dataset in SQLite / Pandas, 
#                plot accuracy vs signal-to-noise 
#                in Matplotlib

###############################################################
###############################################################
# Additional Info:
#
# Google Drive / gridviewer.db was generated 
# using the following code:
# Runtime: ~= 51 minutes
###############################################################
# import grid_search as gs
#parameter_space_subset = {
    #"candidates_per_band": [6,7,8,9,10,20,40],
    #"cm_window_size": [5, 10, 20, 40],
    #"fanout_t": [30, 60, 100],
    #"fanout_f": [500, 1000, 1500, 2000]
#}
#signal_to_noise_ratios = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
#grid_viewer = gs.run_grid_search(parameter_space_subset, signal_to_noise_ratios, n_songs=4)

###############################################################
# note: accuracy is computed using simulated microphone samples
#       view samples with given signal-to-noise ratio with the
#       following function:
# samples = gs.sample_from_source_audios(n_songs, snr=signal_to_noise_ratio)
###############################################################
```



---

# Old TODOs for reference:

## Week 3:

1. Complete TODOs in const_map.py to implement our functions for constellation mapping
2. Visualize the same audio file using cm_visualizations.py and compare it to our solution. Are they similar?

Note: If using WSL: Use explorer.exe <\_.html> to visualize your spectrogram peaks.

Solutions are in test_spec\_"audio_name".html

3. Complete TODOs in hasher.py to implement our function for fingerprinting

4. Use test_hash.py to test your hash creation. Check pb_short_hashed.txt for answers/


## Week 4:

1. Download the week3tracks.zip files listed below (1GB so be warned)
2. Complete TODOs in hasher.py to implement our function for fingerprinting
3. Investigate predict_song.py. Try to understand how data is being passed into the Flask app.
4. Run test_search.py with predict_song.py running in the background to test your audio search
   algorithm. Replace pre-existing audio_path with any song in ./tracks/audio as a test.

Note: you will need to pip install flask and flask-cors to run the test scripts.