import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
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

const MODEL_VERSION = "1.5.0";
const API_BASE = "http://localhost:8000";

const Waveform = () => (
  <div className="flex items-end gap-1 h-6 mt-1">
    {[...Array(5)].map((_, i) => (
      <motion.div
        key={i}
        animate={{ height: ["6px", "18px", "6px"] }}
        transition={{
          repeat: Infinity,
          duration: 1,
          delay: i * 0.15,
        }}
        className="w-1 bg-indigo-500 rounded"
      />
    ))}
  </div>
);

function App() {
  const [text, setText] = useState("");
  const [emotion, setEmotion] = useState("");
  const [songs, setSongs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(null);

  const audioRef = useRef(null);

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
    setCurrentIndex(null);

    try {
      const response = await axios.post(`${API_BASE}/recommend`, { text });

      setEmotion(response.data.predicted_emotion);
      setSongs(response.data.songs);

      setHistory([
        { text, emotion: response.data.predicted_emotion },
        ...history.slice(0, 9),
      ]);
    } catch (error) {
      console.error("Request failed:", error);
      alert("Request failed. Check backend console.");
    }

    setLoading(false);
  };

  const confidence = songs[0] ? songs[0].similarity * 100 : 0;

  const chartData = history.reduce((acc, item) => {
    const existing = acc.find((e) => e.emotion === item.emotion);
    if (existing) existing.count++;
    else acc.push({ emotion: item.emotion, count: 1 });
    return acc;
  }, []);

  const playTrack = (index) => {
    const song = songs[index];
    if (!song.preview_url) return;

    if (audioRef.current) {
      audioRef.current.src = song.preview_url;
      audioRef.current.play().catch(() => {});
      setCurrentIndex(index);
    }
  };

  const playAll = () => {
    if (songs.length > 0) {
      playTrack(0);
    }
  };

  const handleEnded = () => {
    if (currentIndex !== null && currentIndex + 1 < songs.length) {
      playTrack(currentIndex + 1);
    } else {
      setCurrentIndex(null);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0f17] text-white overflow-x-hidden relative font-sans">
      {/* Dynamic Ambient Glow */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-[-250px] left-[-200px] w-[700px] h-[700px] bg-indigo-600/20 blur-[180px] rounded-full" />
        <div className="absolute bottom-[-250px] right-[-200px] w-[700px] h-[700px] bg-purple-600/20 blur-[180px] rounded-full" />
      </div>

      <audio ref={audioRef} onEnded={handleEnded} hidden />

      {/* HEADER */}
      <header className="relative z-10 h-20 px-20 flex items-center justify-between border-b border-white/10 backdrop-blur-xl bg-white/5">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-xl" />
          <div>
            <div className="font-semibold tracking-tight text-lg">Auris</div>
            <div className="text-xs text-gray-400 tracking-wide">
              Music Intelligence
            </div>
          </div>
        </div>

        <div className="text-sm text-gray-400">Model v{MODEL_VERSION}</div>
      </header>

      {/* MAIN */}
      <main className="relative z-10 max-w-6xl mx-auto px-20 py-20">
        {/* HERO */}
        <div className="text-center space-y-6">
          <h1 className="text-6xl font-semibold leading-tight tracking-tight">
            Discover music that
            <br />
            <span className="bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">
              understands you
            </span>
          </h1>

          <p className="text-gray-400 text-lg max-w-2xl mx-auto leading-relaxed">
            Real-time emotional analysis transforms your words into immersive
            music recommendations.
          </p>
        </div>

        {/* INPUT SURFACE */}
        <div className="mt-16 max-w-3xl mx-auto bg-white/5 border border-white/10 backdrop-blur-2xl rounded-3xl p-10 shadow-2xl space-y-8">
          <textarea
            className="w-full bg-transparent border border-white/10 rounded-2xl p-6 focus:outline-none focus:ring-2 focus:ring-indigo-500 placeholder-gray-500 text-lg transition"
            rows="4"
            placeholder="How are you feeling today?"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <div className="flex items-center justify-between">
            <div className="space-y-4">
              {emotion && (
                <div className="inline-flex px-5 py-2 rounded-full bg-indigo-500/20 text-indigo-300 text-sm backdrop-blur-md">
                  {emotion}
                </div>
              )}

              {confidence > 0 && (
                <div className="w-72">
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

        {/* RESULTS STAGE */}
        <div className="mt-24 space-y-10">
          {songs.length === 0 && !loading && (
            <div className="h-64 flex items-center justify-center text-gray-500 border border-white/10 rounded-3xl bg-white/5 backdrop-blur-xl">
              Your music will appear here.
            </div>
          )}

          {songs.map((song, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.08 }}
              className="bg-white/5 border border-white/10 backdrop-blur-xl rounded-3xl p-8 shadow-xl hover:bg-white/10 transition"
            >
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
                {/* Album */}
                <div className="lg:col-span-1">
                  <div className="aspect-square rounded-2xl overflow-hidden bg-white/10">
                    {song.album_image && (
                      <img
                        src={song.album_image}
                        className="w-full h-full object-cover"
                      />
                    )}
                  </div>
                </div>

                {/* Info */}
                <div className="lg:col-span-4 space-y-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="text-2xl font-semibold tracking-tight">
                        {song.title}
                      </h3>
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
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </main>
    </div>
  );
}

export default App;
