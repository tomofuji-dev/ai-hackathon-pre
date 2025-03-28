import { useState } from "react";
import { API_DOMAIN } from "@/const";
import { AxiosClient } from "@/lib/axios";

export interface AudioApiData {
  score: number;
  label: string;
}

export const useAudio = () => {
  const [data, setData] = useState<AudioApiData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const axiosClient = new AxiosClient();

  const uploadAudio = async (file: File): Promise<void> => {
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    const result = await axiosClient.post<AudioApiData, FormData>(
      `${API_DOMAIN}/audio`,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    if (result.ok) {
      const response = result.data;
      setData(response.data);
    } else {
      setError(result.error.message);
    }
    setLoading(false);
  };

  return { data, loading, error, uploadAudio };
};
