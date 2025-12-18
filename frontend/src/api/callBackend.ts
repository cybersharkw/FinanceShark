import { z } from 'zod';
import api from './api';
import axios, { AxiosError } from 'axios';

// Schema to validate that response is a string
const BackendResponseSchema = z.object({
  response: z.string(),
  session_id: z.string(),
  status: z.string()
});

export type BackendResponse = z.infer<typeof BackendResponseSchema>;

/**
 * Calls backend API with POST request
 * @param endpoint - API endpoint path
 * @param data - Data object to send (will be automatically stringified by axios)
 * @returns Promise with string response
 */
export async function callBackend(
  endpoint: string,
  data: Record<string, any>
): Promise<BackendResponse> {
  try {
    const response = await api.post(endpoint, data);
    
    // Validate that response.data is a string
   const validatedData = BackendResponseSchema.parse(response.data);
    
    return validatedData;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      console.error('API Error:', axiosError.message);
      
      if (axiosError.response) {
        console.error('Response status:', axiosError.response.status);
        console.error('Response data:', axiosError.response.data);
      }
    } else {
      console.error('API Error:', error);
    }
    
    throw error;
  }
}