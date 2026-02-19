import { useState, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import CountUp from "react-countup";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const MODEL_VERSION = "4.2.0";
const API_BASE = "http://localhost:8000";

function App() {
  const [text, setText] = useState("");
  const [emotion, setEmotion] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [songs, setSongs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const saved = localStorage.getItem("auris-history");
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  useEffect(() => {
    localStorage.setItem("auris-history", JSON.stringify(history));
  }, [history]);

  const getRecommendations = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setSongs([]);

    try {
      const response = await axios.post(`${API_BASE}/recommend`, { text });

      const data = response.data;

      setEmotion(data.predicted_emotion);
      setConfidence((data.confidence || 0) * 100);
      setSongs(data.songs || []);

      setHistory([
        { text, emotion: data.predicted_emotion },
        ...history.slice(0, 9),
      ]);
    } catch (error) {
      console.error("Backend error:", error);
      alert("Backend error. Check Flask console.");
    }

    setLoading(false);
  };

  const chartData = history.reduce((acc, item) => {
    const existing = acc.find((e) => e.emotion === item.emotion);
    if (existing) existing.count++;
    else acc.push({ emotion: item.emotion, count: 1 });
    return acc;
  }, []);

  return (
    <div className="min-h-screen bg-[#0b0f17] text-white overflow-x-hidden font-sans relative">
      {/* Ambient Glow */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-[-200px] left-[-200px] w-[600px] h-[600px] bg-indigo-600/20 blur-[150px] rounded-full" />
        <div className="absolute bottom-[-200px] right-[-200px] w-[600px] h-[600px] bg-purple-600/20 blur-[150px] rounded-full" />
      </div>

      {/* HEADER */}
      <header className="relative z-10 h-20 px-20 flex items-center justify-between border-b border-white/10 backdrop-blur-xl bg-white/5">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-xl" />
          <div>
            <div className="font-semibold text-lg">Auris</div>
            <div className="text-xs text-gray-400">Music Intelligence</div>
          </div>
        </div>
        <div className="text-sm text-gray-400">Model v{MODEL_VERSION}</div>
      </header>

      <main className="relative z-10 max-w-6xl mx-auto px-20 py-20">
        {/* HERO */}
        <div className="text-center space-y-6">
          <h1 className="text-6xl font-semibold leading-tight">
            Discover music that
            <br />
            <span className="bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">
              understands you
            </span>
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Emotional AI powered recommendations in real-time.
          </p>
        </div>

        {/* INPUT CARD */}
        <div className="mt-16 max-w-3xl mx-auto bg-white/5 border border-white/10 rounded-3xl p-10 shadow-2xl space-y-8 backdrop-blur-xl">
          <textarea
            rows="4"
            placeholder="How are you feeling today?"
            className="w-full bg-transparent border border-white/10 rounded-2xl p-6 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-lg"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <div className="flex items-center justify-between">
            <div>
              {emotion && (
                <div className="inline-flex px-5 py-2 rounded-full bg-indigo-500/20 text-indigo-300 text-sm">
                  {emotion}
                </div>
              )}

              {confidence > 0 && (
                <div className="w-72 mt-4">
                  <div className="text-xs text-gray-400 mb-2">
                    Model Confidence
                  </div>

                  <div className="flex items-center gap-4">
                    <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${confidence}%` }}
                        transition={{ duration: 0.8 }}
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-600"
                      />
                    </div>

                    <div className="text-indigo-400 font-medium">
                      <CountUp end={confidence} duration={1} decimals={1} />%
                    </div>
                  </div>
                </div>
              )}
            </div>

            <button
              onClick={getRecommendations}
              className="px-10 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl font-medium shadow-xl hover:scale-[1.05] transition"
            >
              {loading ? "Analyzing..." : "Generate"}
            </button>
          </div>
        </div>

        {/* RESULTS */}
        <div className="mt-24 space-y-10">
          {songs.length === 0 && !loading && (
            <div className="h-64 flex items-center justify-center text-gray-500 border border-white/10 rounded-3xl bg-white/5">
              Your music will appear here.
            </div>
          )}

          {songs.map((song, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.08 }}
              className="bg-white/5 border border-white/10 rounded-3xl p-8 shadow-xl"
            >
              <div className="flex justify-between">
                <div>
                  <h3 className="text-2xl font-semibold">{song.title}</h3>
                  <p className="text-gray-400 mt-1">{song.artist}</p>
                </div>
                <div className="text-indigo-400 font-medium">
                  {(song.similarity * 100).toFixed(1)}%
                </div>
              </div>

              {song.youtube_embed && (
                <div className="rounded-2xl overflow-hidden mt-6">
                  <iframe
                    className="w-full h-72"
                    src={`${song.youtube_embed}?rel=0`}
                    allowFullScreen
                  />
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* HISTORY CHART */}
        {chartData.length > 0 && (
          <div className="mt-20 bg-white/5 border border-white/10 rounded-3xl p-10">
            <h2 className="text-xl mb-6">Emotion History</h2>

            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="emotion" stroke="#aaa" />
                <YAxis stroke="#aaa" />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
