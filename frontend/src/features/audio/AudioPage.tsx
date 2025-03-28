import { useRecordAudio } from "./hooks/useRecordAudio";
import { useUploadAudio } from "./hooks/useUploadAudio";
import Button from "@mui/material/Button";

function AudioPage() {
  const { audioBlob, isRecording, startRecording, stopRecording } =
    useRecordAudio();
  const { data, loading, error, uploadAudio } = useUploadAudio();

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
      {!isRecording ? (
        <Button variant="contained" onClick={startRecording} disabled={loading}>
          録音開始
        </Button>
      ) : (
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

export default AudioPage;
