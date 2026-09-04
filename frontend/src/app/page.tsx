"use client";

import { useState, useEffect } from "react";

export default function Home() {
  const [tfFile, setTfFile] = useState<File | null>(null);
  const [tfPreview, setTfPreview] = useState<string | null>(null);
  const [tfLoading, setTfLoading] = useState<boolean>(false);
  const [tfResult, setTfResult] = useState<{ label: string; confidence: number } | null>(null);

  const [torchFile, setTorchFile] = useState<File | null>(null);
  const [torchPreview, setTorchPreview] = useState<string | null>(null);
  const [torchLoading, setTorchLoading] = useState<boolean>(false);
  const [torchResult, setTorchResult] = useState<{ label: string; confidence: number }[] | null>(null);

  const [budget, setBudget] = useState<number | string>(100);
  const [screens, setScreens] = useState<number | string>(1200);
  const [regLoading, setRegLoading] = useState<boolean>(false);
  const [regResult, setRegResult] = useState<{ audience: number; image: string } | null>(null);

  const [inputRating, setInputRating] = useState<number>(5);
  const [ratingData, setRatingData] = useState<{ average: number; count: number; chart: string } | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/rating/status")
      .then((res) => res.json())
      .then((data) => setRatingData({ average: data.average, count: data.count, chart: data.chart_image }))
      .catch(() => {});
  }, []);

  const handleTfChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setTfFile(file);
      setTfPreview(URL.createObjectURL(file));
      setTfResult(null);
    }
  };

  const handleTorchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setTorchFile(file);
      setTorchPreview(URL.createObjectURL(file));
      setTorchResult(null);
    }
  };

  const predictTf = async () => {
    if (!tfFile) return;
    setTfLoading(true);
    setTfResult(null);

    const formData = new FormData();
    formData.append("file", tfFile);

    try {
      const res = await fetch("http://localhost:8000/predict/tf", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("분석 실패");
      const data = await res.json();
      setTfResult(data);
    } catch {
      alert("FastAPI 서버(8000)를 확인해주세요!");
    } finally {
      setTfLoading(false);
    }
  };

  const predictTorch = async () => {
    if (!torchFile) return;
    setTorchLoading(true);
    setTorchResult(null);

    const formData = new FormData();
    formData.append("file", torchFile);

    try {
      const res = await fetch("http://localhost:8000/predict/torch", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("분석 실패");
      const data = await res.json();
      setTorchResult(data.results);
    } catch {
      alert("FastAPI 서버(8000)를 확인해주세요!");
    } finally {
      setTorchLoading(false);
    }
  };

  const predictBoxOffice = async () => {
    if (budget === "" || screens === "") {
      alert("제작비와 스크린 수를 모두 입력해주세요.");
      return;
    }

    setRegLoading(true);
    try {
      const res = await fetch("http://localhost:8000/predict/box-office-3d", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ budget: Number(budget), screens: Number(screens) }),
      });
      if (!res.ok) throw new Error("분석 실패");
      const data = await res.json();
      setRegResult({ audience: data.predicted_audience, image: data.graph_image });
    } catch {
      alert("FastAPI 서버(8000)를 확인해주세요!");
    } finally {
      setRegLoading(false);
    }
  };

  const submitRating = async () => {
    try {
      const res = await fetch("http://localhost:8000/rating/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating: Number(inputRating) }),
      });
      if (!res.ok) throw new Error("등록 실패");
      const data = await res.json();
      setRatingData({ average: data.average, count: data.count, chart: data.chart_image });
    } catch {
      alert("FastAPI 서버를 확인해주세요.");
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 py-12 px-6">
      <div className="max-w-6xl mx-auto space-y-8">
        <header className="text-center">
          <h1 className="text-3xl font-bold text-slate-800">
            🧠 AI 머신러닝 & 딥러닝 분류/회귀
          </h1>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <span className="text-2xl">🐾</span>
                <div>
                  <h2 className="text-lg font-bold text-slate-800">개 vs 고양이 분류기</h2>
                  <p className="text-xs text-blue-600 font-semibold">Engine: TensorFlow (Keras)</p>
                </div>
              </div>

              <div className="border-2 border-dashed border-slate-200 rounded-xl p-4 flex flex-col items-center justify-center bg-slate-50 min-h-[220px] mb-4">
                {tfPreview ? (
                  <img src={tfPreview} alt="개/고양이" className="max-h-44 object-contain rounded-lg shadow-sm" />
                ) : (
                  <p className="text-sm text-slate-400">강아지나 고양이 사진을 올려주세요</p>
                )}
              </div>

              <input
                type="file"
                accept="image/*"
                onChange={handleTfChange}
                className="w-full text-sm text-slate-500 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer mb-4"
              />

              <button
                onClick={predictTf}
                disabled={!tfFile || tfLoading}
                className={`w-full py-2.5 rounded-xl font-bold text-white transition ${
                  !tfFile || tfLoading ? "bg-slate-300 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
                }`}
              >
                {tfLoading ? "분석 중..." : "TensorFlow로 분석하기"}
              </button>
            </div>

            <div className="mt-6 pt-4 border-t border-slate-100 min-h-[90px]">
              <span className="text-xs font-semibold text-slate-400 block mb-2">분석 결과</span>
              {tfResult ? (
                <div className="flex items-center justify-between p-3 rounded-lg bg-blue-50 text-blue-900">
                  <span className="font-bold text-base">{tfResult.label}</span>
                  <span className="text-sm font-semibold text-blue-600">{tfResult.confidence}%</span>
                </div>
              ) : (
                <p className="text-xs text-slate-400 italic">이미지를 업로드하고 분석을 실행하세요.</p>
              )}
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <span className="text-2xl">🔍</span>
                <div>
                  <h2 className="text-lg font-bold text-slate-800">사물 1,000종 감지기</h2>
                  <p className="text-xs text-indigo-600 font-semibold">Engine: PyTorch (MobileNetV2)</p>
                </div>
              </div>

              <div className="border-2 border-dashed border-slate-200 rounded-xl p-4 flex flex-col items-center justify-center bg-slate-50 min-h-[220px] mb-4">
                {torchPreview ? (
                  <img src={torchPreview} alt="사물" className="max-h-44 object-contain rounded-lg shadow-sm" />
                ) : (
                  <p className="text-sm text-slate-400">일상 사물 사진을 올려주세요</p>
                )}
              </div>

              <input
                type="file"
                accept="image/*"
                onChange={handleTorchChange}
                className="w-full text-sm text-slate-500 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer mb-4"
              />

              <button
                onClick={predictTorch}
                disabled={!torchFile || torchLoading}
                className={`w-full py-2.5 rounded-xl font-bold text-white transition ${
                  !torchFile || torchLoading ? "bg-slate-300 cursor-not-allowed" : "bg-indigo-600 hover:bg-indigo-700"
                }`}
              >
                {torchLoading ? "분석 중..." : "PyTorch로 분석하기"}
              </button>
            </div>

            <div className="mt-6 pt-4 border-t border-slate-100 min-h-[90px]">
              <span className="text-xs font-semibold text-slate-400 block mb-2">분석 결과 (Top 3)</span>
              {torchResult ? (
                <ul className="space-y-1.5 p-2 rounded-lg bg-indigo-50 text-indigo-900">
                  {torchResult.map((item, idx) => (
                    <li key={idx} className="flex justify-between items-center text-xs font-medium">
                      <span>{idx + 1}. {item.label}</span>
                      <span className="font-semibold text-indigo-600">{item.confidence}%</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-slate-400 italic">이미지를 업로드하고 분석을 실행하세요.</p>
              )}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
          <div className="flex items-center gap-2 mb-6">
            <span className="text-2xl">📊</span>
            <div>
              <h2 className="text-lg font-bold text-slate-800">영화 흥행(누적 관객 수) 3D 다중 선형 회귀</h2>
              <p className="text-xs text-emerald-600 font-semibold">Engine: Scikit-Learn (Multiple Linear Regression)</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
            <div className="space-y-4">
              <div>
                <label className="text-xs font-semibold text-slate-600 block mb-1">순 제작비 (억 원)</label>
                <input
                  type="number"
                  value={budget}
                  onChange={(e) => setBudget(e.target.value === "" ? "" : Number(e.target.value))}
                  className="w-full border border-slate-200 rounded-lg p-2.5 text-sm focus:outline-blue-500"
                  placeholder="예: 120"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-600 block mb-1">스크린 수 (개봉관 수)</label>
                <input
                  type="number"
                  value={screens}
                  onChange={(e) => setScreens(e.target.value === "" ? "" : Number(e.target.value))}
                  className="w-full border border-slate-200 rounded-lg p-2.5 text-sm focus:outline-blue-500"
                  placeholder="예: 1500"
                />
              </div>

              <button
                onClick={predictBoxOffice}
                disabled={regLoading}
                className={`w-full py-2.5 rounded-xl font-bold text-white transition ${
                  regLoading ? "bg-slate-300 cursor-not-allowed" : "bg-emerald-600 hover:bg-emerald-700"
                }`}
              >
                {regLoading ? "3D 그래프 계산 중..." : "관객 수 예측 & 3D 그래프 생성"}
              </button>

              {regResult && (
                <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-100">
                  <span className="text-xs font-semibold text-emerald-800 block">예상 누적 관객 수</span>
                  <p className="text-2xl font-black text-emerald-600 mt-1">
                    약 {regResult.audience}만 명
                  </p>
                </div>
              )}
            </div>

            <div className="lg:col-span-2 border-2 border-dashed border-slate-200 rounded-xl p-3 bg-slate-50 min-h-[360px] flex items-center justify-center">
              {regResult ? (
                <img
                  src={regResult.image}
                  alt="3D Regression Plot"
                  className="max-h-[380px] w-auto object-contain rounded-lg shadow-sm"
                />
              ) : (
                <p className="text-sm text-slate-400 italic">조건을 입력하고 버튼을 누르면 3D 회귀 평면 그래프가 생성됩니다.</p>
              )}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
          <div className="flex items-center gap-2 mb-4">
            <span className="text-2xl">⭐</span>
            <div>
              <h2 className="text-lg font-bold text-slate-800">실시간 영화 평점 업데이트</h2>
              <p className="text-xs text-sky-600 font-semibold">Engine: Scikit-Learn (SGDRegressor Online Learning)</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            <div className="space-y-4">
              <div>
                <label className="text-xs font-semibold text-slate-600 block mb-1">
                  영화 A 평점 남기기 (1.0 ~ 5.0)
                </label>
                <select
                  value={inputRating}
                  onChange={(e) => setInputRating(Number(e.target.value))}
                  className="w-full border border-slate-200 rounded-lg p-2.5 text-sm focus:outline-blue-500"
                >
                  <option value={5}>⭐⭐⭐⭐⭐ (5점)</option>
                  <option value={4}>⭐⭐⭐⭐ (4점)</option>
                  <option value={3}>⭐⭐⭐ (3점)</option>
                  <option value={2}>⭐⭐ (2점)</option>
                  <option value={1}>⭐ (1점)</option>
                </select>
              </div>

              <button
                onClick={submitRating}
                className="w-full py-2.5 rounded-xl font-bold text-white bg-sky-500 hover:bg-sky-600 transition"
              >
                평점 등록 & 모델 실시간 갱신
              </button>

              {ratingData && (
                <div className="p-3 bg-sky-50 rounded-xl border border-sky-100 text-xs text-slate-600 space-y-1">
                  <p>누적 평가 수: <strong className="text-sky-700">{ratingData.count}명</strong></p>
                  <p>현재 평균 평점: <strong className="text-sky-700">{ratingData.average}점</strong></p>
                </div>
              )}
            </div>

            <div className="md:col-span-2 flex justify-center items-center bg-slate-50 border-2 border-dashed border-slate-200 rounded-xl p-3 min-h-[260px]">
              {ratingData ? (
                <img src={ratingData.chart} alt="Real-time Rating Chart" className="rounded-lg shadow-sm" />
              ) : (
                <p className="text-xs text-slate-400">데이터를 불러오는 중입니다...</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}