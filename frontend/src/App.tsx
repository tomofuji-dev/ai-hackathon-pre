import { AudioPage } from "@/features/audio";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { HomePage } from "@/features/home";
import { AppLayout } from "@/components/layout";
import { GameProvider } from "@/context/GameContext";

// Create a custom theme with game-like colors
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#6200EA', // Deep purple
    },
    secondary: {
      main: '#00E676', // Green
    },
    background: {
      default: '#121212',
      paper: '#1E1E1E',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
    },
    h2: {
      fontWeight: 600,
    },
    button: {
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          padding: '10px 20px',
          fontSize: '1rem',
        },
        containedPrimary: {
          boxShadow: '0 4px 10px rgba(98, 0, 234, 0.5)',
          '&:hover': {
            boxShadow: '0 6px 15px rgba(98, 0, 234, 0.7)',
          },
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <GameProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<AppLayout />}>
              <Route index element={<HomePage />} />
              <Route path="game" element={<AudioPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </GameProvider>
    </ThemeProvider>
  );
}

export default App;
