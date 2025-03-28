import React, { useState } from "react";
import Button from "@mui/material/Button";
import { useAudio } from "@/hooks/useAudio";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const { data, loading, error, uploadAudio } = useAudio();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleClick = async () => {
    if (!file) {
      alert("ファイルを選択してください");
      return;
    }
    await uploadAudio(file);
  };

  return (
    <div>
      <input type="file" accept="audio/*" onChange={handleFileChange} />
      <Button variant="contained" onClick={handleClick} disabled={loading}>
        {loading ? "送信中..." : "音声ファイルを送信"}
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
