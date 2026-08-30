import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server runs on :5173; it calls the FastAPI backend on :8000 directly
// (the backend allows that origin via CORS).
export default defineConfig({ plugins: [react()] });
