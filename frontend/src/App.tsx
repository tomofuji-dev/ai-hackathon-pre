import React, { useState } from "react";
import Button from "@mui/material/Button";
import { API_DOMAIN } from "./env";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState(null);

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
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_DOMAIN}/wav`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <input type="file" accept="audio/*" onChange={handleFileChange} />
      <Button variant="contained" onClick={handleClick}>
        音声ファイルを送信
      </Button>
      {result && (
        <div>
          <p>{result.label}</p>
        </div>
      )}
    </div>
  );
}

export default App;
