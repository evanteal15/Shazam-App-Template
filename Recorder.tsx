import React, { useEffect, useState, useRef } from "react";
import {
  useAudioRecorder,
  AudioModule,
  useAudioRecorderState,
  setAudioModeAsync,
  RecordingPresets,
  RecordingStatus,
  AudioRecorder,
  useAudioPlayer,
} from "expo-audio";
import { Audio } from "expo-av";
import {
  Button,
  StyleSheet,
  Text,
  TouchableOpacity,
  TextInput,
  View,
  Image,
  Alert,
  Platform,
  KeyboardAvoidingView,
  ActivityIndicator,
} from "react-native";
import YoutubePlayer from "react-native-youtube-iframe";
import { SafeAreaView, SafeAreaProvider } from "react-native-safe-area-context";

function extractYouTubeVideoId(url: string | null): string {
  if (!url) return "";
  const match = url.match(
    /(?:youtube\.com\/.*v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/
  );
  return match ? match[1] : url;
}

export default function Recorder() {
  const [audioInput, setAudioInput] = useState("device");
  const [uri, setUri] = useState<any>(null);
  const [text, setText] = React.useState("");
  const audioRecorder = useAudioRecorder({
    extension: ".m4a",
    numberOfChannels: 2,
    sampleRate: 44100,
    bitRate: 128000,
    android: {
      extension: ".wav",
      outputFormat: "mpeg4",
      audioEncoder: "aac",
      sampleRate: 44100,
    },
    ios: {
      extension: ".wav",
      audioQuality: 2,
      sampleRate: 44100,
      linearPCMBitDepth: 16,
      linearPCMIsBigEndian: false,
      linearPCMIsFloat: false,
    },
    web: {
      mimeType: "audio/wav",
    },
  });
  const recorderState = useAudioRecorderState(audioRecorder);
  const [predictedSong, setPredictedSong] = useState<string | null>(null);
  const [predictedConfidence, setPredictedConfidence] = useState<number | null>(
    null
  );
  const [predictedUrl, setPredictedUrl] = useState<string | null>(null);
  const [showPrediction, setShowPrediction] = useState(false);
  const [addingSong, setAddingSong] = useState(false);
  // const audioPlayer = useAudioPlayer();
  // const [audioPlayer, setAudioPlayer] = useState(null);
  //   const [permission, requestPermission] = useAudPermissions();

  const record = async () => {
    await audioRecorder.prepareToRecordAsync();
    audioRecorder.record();
  };

  const stopRecording = async () => {
    await audioRecorder.stop();
    setUri(audioRecorder.uri);
  };

  useEffect(() => {
    (async () => {
      const status = await AudioModule.getRecordingPermissionsAsync();
      if (!status.granted) {
        Alert.alert(
          "Permission to access microphone must be given to use the app"
        );
      }

      setAudioModeAsync({
        playsInSilentMode: true,
        allowsRecording: true,
      });
    })();
  }, []);

  async function addSongToDatabase(song_url: any) {
    try {
      console.log("Adding song to database:", song_url);

      setAddingSong(true);

      const formData = new FormData();

      // Append the actual file to FormData
      formData.append("youtube_url", song_url);

      // TODO: type in your server address here
      const add_endpoint = "";

      await fetch(add_endpoint, {
        method: "POST",
        body: formData,
      });
      setAddingSong(false);
      Alert.alert("Song added to database!");
    } catch (error) {
      console.error("Error adding song to database:", error);
    }
  }

  const handlePrediction = async () => {
    console.log("Handling prediction...");
    if (uri) {
      // const response = await getPredictedSong();

      const formData = new FormData();

      const responsee = await fetch(uri);
      const blob = await responsee.blob();

      const file = new File([blob], "audio.flac", { type: "audio/flac" });

      console.log("Audio file being sent:", file);

      // Append the actual file to FormData
      formData.append("audio", file);

      console.log("Sending POST request to server...");

      // TODO: type in your server address here
      const predict_endpoint = "";

      fetch(predict_endpoint, {
        method: "POST",
        body: formData,
        // headers: {
        //   "Content-Type": "multipart/form-data",
        // },
      })
        .then((response) => {
          return response.json();
        })
        .then((data) => {
          setPredictedSong(data.best);
          setPredictedConfidence(data.confidence);
          setPredictedUrl(data.urls);
          setShowPrediction(true);
        })
        .catch((error) => {
          console.error("Error in prediction fetch:", error);
        });
    }
  };

  const handleAddSong = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    console.log("Adding song to database:", text);

    setAddingSong(true);

    const formData = new FormData();

    // Append the actual file to FormData
    formData.append("youtube_url", text);

    fetch("http://192.168.1.170:5003/add", {
      method: "POST",
      body: formData,
    })
      .then(() => {
        setAddingSong(false);
        Alert.alert("Song added to database!");
      })
      .catch((error) => {
        console.error("Error adding song to database:", error);
      });
    setText("");
  };

  return (
    <>
      <SafeAreaProvider>
        <SafeAreaView style={styles.container}>
          <View style={{ width: 150, marginBottom: 20, marginTop: 50 }}>
            {/* <Button
              title={
                recorderState.isRecording ? "Stop Recording" : "Start Recording"
              }
              onPress={recorderState.isRecording ? stopRecording : record}
            /> */}
            <TouchableOpacity
              style={styles.recorderButton}
              onPress={recorderState.isRecording ? stopRecording : record}
            >
              {recorderState.isRecording ? (
                <Image
                  source={require("../../assets/images/square-icon.png")}
                  style={{ width: 100, height: 100 }}
                />
              ) : (
                <Image
                  source={require("../../assets/images/free-microphone-icon.png")}
                  style={{ width: 100, height: 100 }}
                />
              )}
            </TouchableOpacity>
          </View>
          <View style={{ width: 150, marginBottom: 20, marginTop: 20 }}>
            <Button
              title="Play Recording"
              onPress={async () => {
                // console.log("Loading sound..");
                const { sound } = await Audio.Sound.createAsync(
                  { uri: uri },
                  { shouldPlay: true }
                );

                // console.log("Playing sound..");
                await sound.playAsync();
              }}
            />
          </View>
          <View style={{ width: 150, marginBottom: 10, marginTop: 20 }}>
            <Button title="Predict Song" onPress={handlePrediction} />
          </View>
          {predictedSong && (
            <View style={{ alignItems: "center", marginTop: 20 }}>
              <Text style={{ fontSize: 15, marginBottom: 20 }}>
                Here is the most likely song from the recorded snippet!
              </Text>
              {/* <Text>Predicted URL: {predictedUrl}</Text> */}
              {/* <YouTube
                    videoId={predictedUrl} // Extract this from your YouTube URL
                    opts={opts}
                    onReady={(event) => event.target.pauseVideo()}
                  /> */}
              <YoutubePlayer
                height={180}
                scale={4}
                play={false}
                videoId={extractYouTubeVideoId(predictedUrl)}
                style={{ alignSelf: "stretch", marginTop: 20 }}
              />
              <Text>If this does not look correct:</Text>
              <Text>1: Please try again with a new clip.</Text>
              <Text>2: Add the song to our database below.</Text>
            </View>
          )}
          <KeyboardAvoidingView
            behavior={Platform.OS === "ios" ? "padding" : "height"} // "padding" works best on iOS
            keyboardVerticalOffset={0} // adjust if you have headers/navbars
          >
            <TextInput
              style={{
                width: "100%",
                alignSelf: "center",
                marginBottom: 10,
                marginTop: 20,
                backgroundColor: "#e0e0e0",
                padding: 10,
                borderRadius: 5,
              }}
              value={text}
              onChangeText={setText}
              placeholder="Enter YouTube URL here..."
              // style={styles.input}
              // Optionally submit on return key:
              onSubmitEditing={(e) => {
                e.preventDefault();
                addSongToDatabase(text);
                setText("");
              }}
              returnKeyType="done"
            />
          </KeyboardAvoidingView>
          {addingSong && (
            <View style={{ width: 150, marginBottom: 20, marginTop: 10 }}>
              <ActivityIndicator size="small" color="#0000ff" />
            </View>
          )}
          {!addingSong && (
            <View style={{ width: 150, marginBottom: 20, marginTop: 10 }}>
              <Button
                title="Add Song"
                onPress={(e) => {
                  e.preventDefault();
                  addSongToDatabase(text);
                  setText("");
                }}
              />
            </View>
          )}
        </SafeAreaView>
      </SafeAreaProvider>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    marginHorizontal: 20,
    backgroundColor: "#ffffffea",
  },
  button: {
    backgroundColor: "#539eb2ff",
    padding: 15,
    borderRadius: 10,
    marginVertical: 10,
  },
  recorderButton: {
    backgroundColor: "#539eb2ff",
    padding: 15,
    width: 150,
    height: 150,
    borderRadius: 100,
    marginVertical: 10,
    alignItems: "center",
    justifyContent: "center",
  },
});
