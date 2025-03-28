import { AudioPage } from "@/features/audio";
import { BrowserRouter, Routes, Route } from "react-router-dom";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/audio" element={<AudioPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
