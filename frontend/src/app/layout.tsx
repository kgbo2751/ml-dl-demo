import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "딥러닝 분류 및 예측 데모",
  description: "Next.js + FastAPI + TensorFlow/PyTorch",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}