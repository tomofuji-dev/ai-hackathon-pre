import { Outlet } from "react-router-dom";
import { Box, Container, AppBar, Toolbar, Typography, Button } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
import { useGame } from "@/context/GameContext";

export const AppLayout = () => {
  const { totalScore, level } = useGame();

  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <AppBar position="static" sx={{ mb: 2 }}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            <Button
              component={RouterLink}
              to="/"
              color="inherit"
              sx={{ fontSize: "1.25rem", fontWeight: "bold" }}
            >
              Voice Master
            </Button>
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <Typography variant="body1" sx={{ fontWeight: "bold" }}>
              Level: {level}
            </Typography>
            <Typography variant="body1" sx={{ fontWeight: "bold" }}>
              Score: {totalScore}
            </Typography>
            <Button
              component={RouterLink}
              to="/game"
              variant="contained"
              color="secondary"
            >
              Play Game
            </Button>
          </Box>
        </Toolbar>
      </AppBar>

      <Container component="main" sx={{ flexGrow: 1, py: 4 }}>
        <Outlet />
      </Container>

      <Box
        component="footer"
        sx={{
          py: 3,
          px: 2,
          mt: "auto",
          backgroundColor: (theme) => theme.palette.background.paper,
          borderTop: (theme) => `1px solid ${theme.palette.divider}`,
        }}
      >
        <Container maxWidth="sm">
          <Typography variant="body2" color="text.secondary" align="center">
            Voice Master Game © {new Date().getFullYear()}
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};
