"use client";

import { useState, useEffect } from "react";

interface YouTubeVideo {
  title: string;
  description: string;
  videoId: string;
}

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

  const [ytQuery, setYtQuery] = useState<string>("");
  const [ytInclude, setYtInclude] = useState<string>("");
  const [ytExclude, setYtExclude] = useState<string>("");
  const [ytUseMl, setYtUseMl] = useState<boolean>(false);
  const [ytLoading, setYtLoading] = useState<boolean>(false);
  const [ytVideos, setYtVideos] = useState<YouTubeVideo[]>([]);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/rating/status")
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
      const res = await fetch("http://127.0.0.1:8000/predict/tf", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("분석 실패");
      const data = await res.json();
      setTfResult(data);
    } catch {
      alert("백엔드 서버(8000)를 확인해주세요!");
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
      const res = await fetch("http://127.0.0.1:8000/predict/torch", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("분석 실패");
      const data = await res.json();
      setTorchResult(data.results);
    } catch {
      alert("백엔드 서버(8000)를 확인해주세요!");
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
      const res = await fetch("http://127.0.0.1:8000/predict/box-office-3d", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ budget: Number(budget), screens: Number(screens) }),
      });
      if (!res.ok) throw new Error("분석 실패");
      const data = await res.json();
      setRegResult({ audience: data.predicted_audience, image: data.graph_image });
    } catch {
      alert("백엔드 서버(8000)를 확인해주세요!");
    } finally {
      setRegLoading(false);
    }
  };

  const submitRating = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/rating/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating: Number(inputRating) }),
      });
      if (!res.ok) throw new Error("등록 실패");
      const data = await res.json();
      setRatingData({ average: data.average, count: data.count, chart: data.chart_image });
    } catch {
      alert("백엔드 서버를 확인해주세요.");
    }
  };

  const searchYouTube = async () => {
    if (!ytQuery.trim()) {
      alert("검색어를 입력해주세요.");
      return;
    }

    setYtLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/youtube/filter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: ytQuery,
          include_kw: ytInclude,
          exclude_kw: ytExclude,
          use_ml: ytUseMl,
        }),
      });
      if (!res.ok) throw new Error("필터링 실패");
      const data = await res.json();
      setYtVideos(data.videos);
    } catch {
      alert("백엔드 서버(8000) 또는 API Key를 확인해주세요!");
    } finally {
      setYtLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-100/70 py-14 px-4 sm:px-8 lg:px-12">
      <div className="max-w-7xl mx-auto space-y-12">
        <header className="text-center space-y-2 pb-4">
          <h1 className="text-3xl md:text-5xl font-black tracking-tight text-slate-800">
            🧠 AI 머신러닝 & 딥러닝 통합 플랫폼
          </h1>
          <p className="text-slate-500 font-medium text-base">
            TensorFlow · PyTorch · Scikit-Learn · YouTube Data API
          </p>
        </header>

        {/* 1. 개 vs 고양이 분류기 */}
        <section className="bg-white rounded-3xl shadow-sm border border-slate-200/90 p-8 lg:p-10">
          <div className="flex items-center gap-3.5 mb-8 pb-4 border-b border-slate-100">
            <span className="text-4xl">🐾</span>
            <div>
              <h2 className="text-2xl font-bold text-slate-800">개 vs 고양이 이미지 분류기</h2>
              <p className="text-sm font-semibold text-blue-600">Engine: TensorFlow (Keras MobileNetV2)</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
            <div className="lg:col-span-5 flex flex-col justify-between space-y-6">
              <div className="space-y-4">
                <label className="text-xs font-bold text-slate-500 tracking-wider uppercase block">
                  이미지 파일 업로드
                </label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleTfChange}
                  className="w-full text-sm text-slate-500 file:mr-4 file:py-3 file:px-6 file:rounded-2xl file:border-0 file:text-xs file:font-bold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"
                />
                <button
                  onClick={predictTf}
                  disabled={!tfFile || tfLoading}
                  className={`w-full py-4 rounded-2xl font-bold text-base text-white transition shadow-sm ${
                    !tfFile || tfLoading ? "bg-slate-300 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700 active:scale-[0.99]"
                  }`}
                >
                  {tfLoading ? "분석 모델 구동 중..." : "TensorFlow로 분석하기"}
                </button>
              </div>

              <div className="pt-6 border-t border-slate-100">
                <span className="text-xs font-bold text-slate-400 tracking-wider uppercase block mb-3">판별 결과</span>
                {tfResult ? (
                  <div className="flex items-center justify-between p-5 rounded-2xl bg-blue-50/80 border border-blue-100 text-blue-950">
                    <span className="font-extrabold text-2xl">{tfResult.label}</span>
                    <span className="text-lg font-bold text-blue-600">신뢰도 {tfResult.confidence}%</span>
                  </div>
                ) : (
                  <div className="p-6 rounded-2xl bg-slate-50 border-2 border-dashed border-slate-200 text-center text-sm text-slate-400">
                    이미지를 업로드하고 분석 버튼을 눌러주세요.
                  </div>
                )}
              </div>
            </div>

            <div className="lg:col-span-7 border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50/70 p-6 min-h-[440px] flex items-center justify-center">
              {tfPreview ? (
                <img src={tfPreview} alt="개/고양이 미리보기" className="max-h-[420px] w-auto object-contain rounded-xl shadow-md" />
              ) : (
                <div className="text-center text-slate-400 space-y-3">
                  <span className="text-5xl block">🖼️</span>
                  <p className="text-base font-medium">대형 미리보기가 여기에 표시됩니다</p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* 2. 사물 1,000종 감지기 */}
        <section className="bg-white rounded-3xl shadow-sm border border-slate-200/90 p-8 lg:p-10">
          <div className="flex items-center gap-3.5 mb-8 pb-4 border-b border-slate-100">
            <span className="text-4xl">🔍</span>
            <div>
              <h2 className="text-2xl font-bold text-slate-800">사물 1,000종 이미지 감지기</h2>
              <p className="text-sm font-semibold text-indigo-600">Engine: PyTorch (MobileNetV2 ImageNet)</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
            <div className="lg:col-span-5 flex flex-col justify-between space-y-6">
              <div className="space-y-4">
                <label className="text-xs font-bold text-slate-500 tracking-wider uppercase block">
                  이미지 파일 업로드
                </label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleTorchChange}
                  className="w-full text-sm text-slate-500 file:mr-4 file:py-3 file:px-6 file:rounded-2xl file:border-0 file:text-xs file:font-bold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer"
                />
                <button
                  onClick={predictTorch}
                  disabled={!torchFile || torchLoading}
                  className={`w-full py-4 rounded-2xl font-bold text-base text-white transition shadow-sm ${
                    !torchFile || torchLoading ? "bg-slate-300 cursor-not-allowed" : "bg-indigo-600 hover:bg-indigo-700 active:scale-[0.99]"
                  }`}
                >
                  {torchLoading ? "분석 모델 구동 중..." : "PyTorch로 분석하기"}
                </button>
              </div>

              <div className="pt-6 border-t border-slate-100">
                <span className="text-xs font-bold text-slate-400 tracking-wider uppercase block mb-3">상위 3위 인식 결과</span>
                {torchResult ? (
                  <ul className="space-y-3 p-4 rounded-2xl bg-indigo-50/80 border border-indigo-100 text-indigo-950">
                    {torchResult.map((item, idx) => (
                      <li key={idx} className="flex justify-between items-center text-sm font-bold p-3 bg-white rounded-xl shadow-xs">
                        <span>{idx + 1}. {item.label}</span>
                        <span className="font-extrabold text-indigo-600">{item.confidence}%</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="p-6 rounded-2xl bg-slate-50 border-2 border-dashed border-slate-200 text-center text-sm text-slate-400">
                    이미지를 업로드하고 분석 버튼을 눌러주세요.
                  </div>
                )}
              </div>
            </div>

            <div className="lg:col-span-7 border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50/70 p-6 min-h-[440px] flex items-center justify-center">
              {torchPreview ? (
                <img src={torchPreview} alt="사물 미리보기" className="max-h-[420px] w-auto object-contain rounded-xl shadow-md" />
              ) : (
                <div className="text-center text-slate-400 space-y-3">
                  <span className="text-5xl block">🖼️</span>
                  <p className="text-base font-medium">대형 미리보기가 여기에 표시됩니다</p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* 3. 영화 관객 수 3D 회귀 */}
        <section className="bg-white rounded-3xl shadow-sm border border-slate-200/90 p-8 lg:p-10">
          <div className="flex items-center gap-3.5 mb-8 pb-4 border-b border-slate-100">
            <span className="text-4xl">📊</span>
            <div>
              <h2 className="text-2xl font-bold text-slate-800">영화 흥행(누적 관객 수) 3D 다중 선형 회귀</h2>
              <p className="text-sm font-semibold text-emerald-600">Engine: Scikit-Learn (Multiple Linear Regression)</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
            <div className="lg:col-span-4 space-y-6">
              <div>
                <label className="text-xs font-bold text-slate-600 block mb-2">순 제작비 (억 원)</label>
                <input
                  type="number"
                  value={budget}
                  onChange={(e) => setBudget(e.target.value === "" ? "" : Number(e.target.value))}
                  className="w-full border border-slate-200 rounded-2xl p-3.5 text-base focus:outline-emerald-500"
                  placeholder="예: 120"
                />
              </div>

              <div>
                <label className="text-xs font-bold text-slate-600 block mb-2">스크린 수 (개봉관 수)</label>
                <input
                  type="number"
                  value={screens}
                  onChange={(e) => setScreens(e.target.value === "" ? "" : Number(e.target.value))}
                  className="w-full border border-slate-200 rounded-2xl p-3.5 text-base focus:outline-emerald-500"
                  placeholder="예: 1500"
                />
              </div>

              <button
                onClick={predictBoxOffice}
                disabled={regLoading}
                className={`w-full py-4 rounded-2xl font-bold text-base text-white transition shadow-sm ${
                  regLoading ? "bg-slate-300 cursor-not-allowed" : "bg-emerald-600 hover:bg-emerald-700 active:scale-[0.99]"
                }`}
              >
                {regLoading ? "3D 그래프 연산 중..." : "관객 수 예측 & 3D 평면 생성"}
              </button>

              {regResult && (
                <div className="p-6 rounded-2xl bg-emerald-50/80 border border-emerald-100 text-center sm:text-left">
                  <span className="text-xs font-bold text-emerald-800 uppercase tracking-wider block">예상 누적 관객 수</span>
                  <p className="text-4xl font-black text-emerald-600 mt-2">
                    약 {regResult.audience}만 명
                  </p>
                </div>
              )}
            </div>

            <div className="lg:col-span-8 border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50/70 p-6 min-h-[500px] flex items-center justify-center">
              {regResult ? (
                <img
                  src={regResult.image}
                  alt="3D Regression Plot"
                  className="max-h-[520px] w-auto object-contain rounded-2xl shadow-md"
                />
              ) : (
                <p className="text-base text-slate-400 italic">조건을 입력하고 예측 버튼을 누르면 3D 회귀 평면 그래프가 생성됩니다.</p>
              )}
            </div>
          </div>
        </section>

        {/* 4. 실시간 영화 평점 업데이트 */}
        <section className="bg-white rounded-3xl shadow-sm border border-slate-200/90 p-8 lg:p-10">
          <div className="flex items-center gap-3.5 mb-8 pb-4 border-b border-slate-100">
            <span className="text-4xl">⭐</span>
            <div>
              <h2 className="text-2xl font-bold text-slate-800">실시간 영화 평점 업데이트 & 온라인 학습</h2>
              <p className="text-sm font-semibold text-sky-600">Engine: Scikit-Learn (SGDRegressor Online Learning)</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
            <div className="lg:col-span-4 space-y-6">
              <div>
                <label className="text-xs font-bold text-slate-600 block mb-2">
                  영화 A 평점 매기기 (1.0 ~ 5.0)
                </label>
                <select
                  value={inputRating}
                  onChange={(e) => setInputRating(Number(e.target.value))}
                  className="w-full border border-slate-200 rounded-2xl p-3.5 text-base focus:outline-sky-500 bg-white"
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
                className="w-full py-4 rounded-2xl font-bold text-base text-white bg-sky-500 hover:bg-sky-600 active:scale-[0.99] transition shadow-sm"
              >
                평점 등록 & 모델 실시간 갱신
              </button>

              {ratingData && (
                <div className="p-6 bg-sky-50/80 rounded-2xl border border-sky-100 text-slate-700 space-y-2">
                  <p className="flex justify-between items-center text-sm font-semibold">
                    <span>누적 평가 수</span>
                    <strong className="text-sky-700 text-lg">{ratingData.count}명</strong>
                  </p>
                  <p className="flex justify-between items-center text-sm font-semibold">
                    <span>실시간 평균 평점</span>
                    <strong className="text-sky-700 text-lg">{ratingData.average}점</strong>
                  </p>
                </div>
              )}
            </div>

            <div className="lg:col-span-8 flex justify-center items-center bg-slate-50/70 border-2 border-dashed border-slate-200 rounded-2xl p-6 min-h-[420px]">
              {ratingData ? (
                <img src={ratingData.chart} alt="Real-time Rating Chart" className="max-h-[400px] w-auto object-contain rounded-2xl shadow-md" />
              ) : (
                <p className="text-base text-slate-400">데이터를 불러오는 중입니다...</p>
              )}
            </div>
          </div>
        </section>

        {/* 5. 유튜브 스마트 필터링 */}
        <section className="bg-white rounded-3xl shadow-sm border border-slate-200/90 p-8 lg:p-10">
          <div className="flex items-center gap-3.5 mb-8 pb-4 border-b border-slate-100">
            <span className="text-4xl">🎥</span>
            <div>
              <h2 className="text-2xl font-bold text-slate-800">유튜브 맞춤 스마트 필터링 시스템</h2>
              <p className="text-sm font-semibold text-rose-600">Engine: Scikit-Learn (TF-IDF + LogisticRegression) & YouTube Data API</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-5 mb-8">
            <div>
              <label className="text-xs font-bold text-slate-600 block mb-2">검색어</label>
              <input
                type="text"
                value={ytQuery}
                onChange={(e) => setYtQuery(e.target.value)}
                placeholder="예: 파이썬 강의"
                className="w-full border border-slate-200 rounded-2xl p-3.5 text-sm focus:outline-rose-500"
              />
            </div>
            <div>
              <label className="text-xs font-bold text-slate-600 block mb-2">포함 키워드 (쉼표 구분)</label>
              <input
                type="text"
                value={ytInclude}
                onChange={(e) => setYtInclude(e.target.value)}
                placeholder="예: 기초, 문법"
                className="w-full border border-slate-200 rounded-2xl p-3.5 text-sm focus:outline-rose-500"
              />
            </div>
            <div>
              <label className="text-xs font-bold text-slate-600 block mb-2">제외 키워드 (쉼표 구분)</label>
              <input
                type="text"
                value={ytExclude}
                onChange={(e) => setYtExclude(e.target.value)}
                placeholder="예: 게임, 쇼츠"
                className="w-full border border-slate-200 rounded-2xl p-3.5 text-sm focus:outline-rose-500"
              />
            </div>
            <div className="flex flex-col justify-end space-y-3">
              <div className="flex items-center gap-2.5 pb-1">
                <input
                  type="checkbox"
                  id="ytMl"
                  checked={ytUseMl}
                  onChange={(e) => setYtUseMl(e.target.checked)}
                  className="w-5 h-5 text-rose-600 rounded-lg focus:ring-rose-500 cursor-pointer"
                />
                <label htmlFor="ytMl" className="text-sm font-bold text-slate-700 cursor-pointer">
                  AI/ML 연관 필터 적용
                </label>
              </div>
              <button
                onClick={searchYouTube}
                disabled={ytLoading}
                className={`w-full py-3.5 rounded-2xl font-bold text-base text-white transition shadow-sm ${
                  ytLoading ? "bg-slate-300 cursor-not-allowed" : "bg-rose-600 hover:bg-rose-700 active:scale-[0.99]"
                }`}
              >
                {ytLoading ? "분석 및 필터링 중..." : "영상 검색 & 필터링"}
              </button>
            </div>
          </div>

          <div className="border-t border-slate-100 pt-8">
            {ytVideos.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
                {ytVideos.map((v) => (
                  <div key={v.videoId} className="bg-slate-50 border border-slate-200 rounded-2xl overflow-hidden p-5 flex flex-col justify-between shadow-sm hover:shadow-md transition">
                    <div>
                      <div className="aspect-video w-full mb-4 rounded-xl overflow-hidden bg-black shadow-inner">
                        <iframe
                          src={`https://www.youtube.com/embed/${v.videoId}`}
                          title={v.title}
                          className="w-full h-full"
                          allowFullScreen
                        />
                      </div>
                      <h3 className="font-bold text-base text-slate-800 line-clamp-2 mb-2 leading-snug">{v.title}</h3>
                      <p className="text-xs text-slate-500 line-clamp-3 leading-relaxed">{v.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-20 border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50">
                <p className="text-base text-slate-400 italic">
                  {ytLoading ? "영상을 분석 및 필터링하고 있습니다..." : "검색 조건을 입력하고 필터링 버튼을 눌러주세요."}
                </p>
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}