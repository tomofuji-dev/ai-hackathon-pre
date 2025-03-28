import { useState } from "react";
import Button from "@mui/material/Button";
import { useAudio } from "@/hooks/useAudio";

function App() {
  const { data, loading, error, uploadAudio } = useAudio();
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(
    null
  );
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert("このブラウザは音声録音をサポートしていません");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        setAudioBlob(blob);
      };
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone", err);
      alert("マイクへのアクセスが拒否されました");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setIsRecording(false);
    }
  };

  const handleClick = async () => {
    if (!audioBlob) {
      alert("録音された音声がありません");
      return;
    }

    const file = new File([audioBlob], "recorded_audio.webm", {
      type: audioBlob.type,
    });
    await uploadAudio(file);
  };

  return (
    <div>
      {!isRecording && (
        <Button variant="contained" onClick={startRecording} disabled={loading}>
          録音開始
        </Button>
      )}
      {isRecording && (
        <Button variant="contained" onClick={stopRecording} disabled={loading}>
          録音停止
        </Button>
      )}
      <Button
        variant="contained"
        onClick={handleClick}
        disabled={loading || !audioBlob}
      >
        {loading ? "送信中..." : "録音音声を送信"}
      </Button>
      {error && (
        <div>
          <p>{error}</p>
        </div>
      )}
      {data && (
        <div>
          <p>{data.label}</p>
          <p>{data.score}</p>
        </div>
      )}
    </div>
  );
}

export default App;
