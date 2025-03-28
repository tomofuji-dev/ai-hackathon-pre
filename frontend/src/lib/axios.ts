import axios, { AxiosRequestConfig, AxiosResponse } from "axios";
import { Result } from "./result";

export class AxiosClient {
  async post<T, U>(
    url: string,
    data?: U,
    config?: AxiosRequestConfig
  ): Promise<Result<AxiosResponse<T>>> {
    try {
      const response = await axios.post<T>(url, data, config);
      return { ok: true, data: response };
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        return { ok: false, error: error };
      } else {
        return { ok: false, error: new Error("Unknown Error At Axios.") };
      }
    }
  }
}
