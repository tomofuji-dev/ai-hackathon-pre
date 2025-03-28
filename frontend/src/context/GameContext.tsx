import { createContext, useContext, useState, ReactNode } from "react";

interface GameScore {
  score: number;
  label: string;
  timestamp: string;
}

interface GameContextType {
  scores: GameScore[];
  addScore: (score: number, label: string) => void;
  totalScore: number;
  level: number;
  resetGame: () => void;
}

const GameContext = createContext<GameContextType | undefined>(undefined);

export const useGame = () => {
  const context = useContext(GameContext);
  if (context === undefined) {
    throw new Error("useGame must be used within a GameProvider");
  }
  return context;
};

interface GameProviderProps {
  children: ReactNode;
}

export const GameProvider = ({ children }: GameProviderProps) => {
  const [scores, setScores] = useState<GameScore[]>([]);

  const addScore = (score: number, label: string) => {
    const newScore = {
      score,
      label,
      timestamp: new Date().toISOString(),
    };
    setScores((prev) => [...prev, newScore]);
  };

  const totalScore = scores.reduce((sum, score) => sum + score.score, 0);
  
  // Calculate level based on total score
  const level = Math.floor(totalScore / 10) + 1;

  const resetGame = () => {
    setScores([]);
  };

  return (
    <GameContext.Provider
      value={{
        scores,
        addScore,
        totalScore,
        level,
        resetGame,
      }}
    >
      {children}
    </GameContext.Provider>
  );
};
