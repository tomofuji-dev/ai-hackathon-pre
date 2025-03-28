import { useEffect, useState, useRef } from "react";
import { useRecordAudio } from "./hooks/useRecordAudio";
import { useUploadAudio } from "./hooks/useUploadAudio";
import { useGame } from "@/context/GameContext";
import { 
  Box, 
  Button, 
  Typography, 
  Paper, 
  CircularProgress, 
  Grid, 
  Card, 
  CardContent,
  Divider,
  Chip,
  Fade,
  Grow,
  LinearProgress,
  Tooltip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from "@mui/material";
import MicIcon from "@mui/icons-material/Mic";
import StopIcon from "@mui/icons-material/Stop";
import SendIcon from "@mui/icons-material/Send";
import HistoryIcon from "@mui/icons-material/History";
import InfoIcon from "@mui/icons-material/Info";
import CloseIcon from "@mui/icons-material/Close";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import { useNavigate } from "react-router-dom";
import { AudioVisualizer } from "./components/AudioVisualizer";
import { ScoreAnimation } from "./components/ScoreAnimation";
import { ResultCard } from "./components/ResultCard";

export const AudioPage = () => {
  const { audioBlob, isRecording, startRecording, stopRecording } = useRecordAudio();
  const { data, loading, error, uploadAudio } = useUploadAudio();
  const { addScore, scores } = useGame();
  const navigate = useNavigate();
  
  const [showResult, setShowResult] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showAnimation, setShowAnimation] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Create audio URL when blob is available
  useEffect(() => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
      return () => {
        URL.revokeObjectURL(url);
      };
    }
  }, [audioBlob]);

  // Handle countdown for recording
  useEffect(() => {
    let timer: number | undefined;
    if (isRecording && countdown === null) {
      setCountdown(5);
      timer = window.setInterval(() => {
        setCountdown((prev) => {
          if (prev === null || prev <= 1) {
            clearInterval(timer);
            stopRecording();
            return null;
          }
          return prev - 1;
        });
      }, 1000);
    }
    
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isRecording, countdown, stopRecording]);

  // Reset result display when starting a new recording
  useEffect(() => {
    if (isRecording) {
      setShowResult(false);
      setShowAnimation(false);
    }
  }, [isRecording]);

  // Add score to game context when data is received
  useEffect(() => {
    if (data && !showResult) {
      addScore(data.score, data.label);
      setShowAnimation(true);
      
      // Show result after animation
      setTimeout(() => {
        setShowResult(true);
        setShowAnimation(false);
      }, 2000);
    }
  }, [data, addScore, showResult]);

  const handleSubmit = async () => {
    if (!audioBlob) {
      return;
    }
    
    const file = new File([audioBlob], "recorded_audio.webm", {
      type: audioBlob.type,
    });
    
    await uploadAudio(file);
  };

  const handlePlayAudio = () => {
    if (audioRef.current && audioUrl) {
      audioRef.current.play();
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "success.main";
    if (score >= 50) return "warning.main";
    return "error.main";
  };

  const getScoreMessage = (score: number, label: string) => {
    if (score >= 80) return `素晴らしい！「${label}」を完璧に発音できています！`;
    if (score >= 50) return `良い調子です！「${label}」の発音がもう少し改善できます。`;
    return `もう一度挑戦してみましょう。「${label}」の発音を練習しましょう。`;
  };

  const getLevelUpMessage = (score: number) => {
    if (score >= 80) return "レベルアップまであと少し！";
    if (score >= 50) return "順調に進んでいます！";
    return "諦めずに続けましょう！";
  };

  return (
    <Box sx={{ maxWidth: 900, mx: "auto", py: 2 }}>
      <audio ref={audioRef} src={audioUrl || ""} style={{ display: "none" }} />
      
      {/* Game Title */}
      <Typography 
        variant="h4" 
        component="h1" 
        gutterBottom 
        align="center"
        sx={{ 
          mb: 4, 
          fontWeight: "bold",
          color: "primary.main"
        }}
      >
        Voice Challenge
      </Typography>
      
      {/* Main Game Area */}
      <Grid container spacing={3}>
        {/* Recording Controls */}
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 3, 
              height: "100%", 
              display: "flex", 
              flexDirection: "column",
              position: "relative",
              overflow: "hidden",
              borderRadius: 2,
              background: "rgba(30, 30, 30, 0.8)",
            }}
          >
            <Typography variant="h6" gutterBottom>
              Record Your Voice
            </Typography>
            
            {/* Audio Visualizer */}
            <Box 
              sx={{ 
                flex: 1, 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "center",
                minHeight: 200,
                mb: 3,
                position: "relative"
              }}
            >
              <AudioVisualizer isRecording={isRecording} />
              
              {countdown !== null && (
                <Typography 
                  variant="h1" 
                  sx={{ 
                    position: "absolute", 
                    color: "primary.main",
                    fontWeight: "bold",
                    opacity: 0.8,
                    fontSize: "5rem"
                  }}
                >
                  {countdown}
                </Typography>
              )}
            </Box>
            
            {/* Controls */}
            <Box sx={{ display: "flex", gap: 2, justifyContent: "center" }}>
              {!isRecording ? (
                <Button 
                  variant="contained" 
                  color="primary"
                  size="large"
                  startIcon={<MicIcon />}
                  onClick={startRecording} 
                  disabled={loading}
                  sx={{ 
                    py: 1.5,
                    px: 3,
                    borderRadius: 8,
                    transition: "transform 0.2s",
                    "&:hover": {
                      transform: "scale(1.05)",
                    },
                  }}
                >
                  録音開始
                </Button>
              ) : (
                <Button 
                  variant="contained" 
                  color="error"
                  size="large"
                  startIcon={<StopIcon />}
                  onClick={stopRecording} 
                  disabled={loading}
                  sx={{ 
                    py: 1.5,
                    px: 3,
                    borderRadius: 8,
                  }}
                >
                  録音停止
                </Button>
              )}
              
              <Button
                variant="contained"
                color="secondary"
                size="large"
                startIcon={<SendIcon />}
                onClick={handleSubmit}
                disabled={loading || !audioBlob || isRecording}
                sx={{ 
                  py: 1.5,
                  px: 3,
                  borderRadius: 8,
                }}
              >
                {loading ? (
                  <>
                    <CircularProgress size={24} color="inherit" sx={{ mr: 1 }} />
                    送信中...
                  </>
                ) : (
                  "送信"
                )}
              </Button>
            </Box>
            
            {audioBlob && !isRecording && (
              <Box sx={{ mt: 2, textAlign: "center" }}>
                <Button 
                  variant="outlined" 
                  size="small" 
                  onClick={handlePlayAudio}
                  sx={{ borderRadius: 4 }}
                >
                  録音を再生
                </Button>
              </Box>
            )}
            
            {error && (
              <Typography color="error" sx={{ mt: 2 }}>
                エラー: {error}
              </Typography>
            )}
          </Paper>
        </Grid>
        
        {/* Results Area */}
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 3, 
              height: "100%", 
              display: "flex", 
              flexDirection: "column",
              position: "relative",
              borderRadius: 2,
              background: "rgba(30, 30, 30, 0.8)",
            }}
          >
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Typography variant="h6">
                Results
              </Typography>
              <Tooltip title="履歴を表示">
                <IconButton onClick={() => setShowHistory(true)} disabled={scores.length === 0}>
                  <HistoryIcon />
                </IconButton>
              </Tooltip>
            </Box>
            
            {showAnimation && data && (
              <ScoreAnimation score={data.score} />
            )}
            
            {showResult && data ? (
              <Grow in={showResult} timeout={800}>
                <Box>
                  <ResultCard 
                    score={data.score} 
                    label={data.label} 
                    scoreColor={getScoreColor(data.score)}
                    message={getScoreMessage(data.score, data.label)}
                    levelMessage={getLevelUpMessage(data.score)}
                  />
                  
                  <Box sx={{ display: "flex", justifyContent: "center", mt: 3 }}>
                    <Button 
                      variant="contained" 
                      color="primary"
                      onClick={() => {
                        setShowResult(false);
                        setCountdown(null);
                      }}
                      sx={{ 
                        borderRadius: 8,
                        px: 3,
                      }}
                    >
                      もう一度挑戦する
                    </Button>
                  </Box>
                </Box>
              </Grow>
            ) : (
              <Box 
                sx={{ 
                  flex: 1, 
                  display: "flex", 
                  flexDirection: "column", 
                  justifyContent: "center", 
                  alignItems: "center",
                  opacity: 0.7,
                }}
              >
                {!loading && !data && (
                  <>
                    <InfoIcon sx={{ fontSize: 60, color: "text.secondary", mb: 2 }} />
                    <Typography variant="body1" color="text.secondary" align="center">
                      録音して送信すると、結果がここに表示されます
                    </Typography>
                  </>
                )}
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
      
      {/* History Dialog */}
      <Dialog 
        open={showHistory} 
        onClose={() => setShowHistory(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Typography variant="h6">
              <HistoryIcon sx={{ mr: 1, verticalAlign: "middle" }} />
              履歴
            </Typography>
            <IconButton onClick={() => setShowHistory(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          {scores.length === 0 ? (
            <Typography align="center" color="text.secondary" sx={{ py: 4 }}>
              まだ履歴がありません
            </Typography>
          ) : (
            <Box sx={{ maxHeight: 400, overflow: "auto" }}>
              {[...scores].reverse().map((score, index) => (
                <Card key={index} sx={{ mb: 2, bgcolor: "background.paper" }}>
                  <CardContent>
                    <Grid container alignItems="center">
                      <Grid item xs={3}>
                        <Typography variant="h5" color={getScoreColor(score.score)}>
                          {score.score}点
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body1">
                          「{score.label}」
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {new Date(score.timestamp).toLocaleString()}
                        </Typography>
                      </Grid>
                      <Grid item xs={3} sx={{ textAlign: "right" }}>
                        {score.score >= 80 && (
                          <Tooltip title="高得点達成！">
                            <EmojiEventsIcon color="secondary" sx={{ fontSize: 30 }} />
                          </Tooltip>
                        )}
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              ))}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowHistory(false)}>閉じる</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};
