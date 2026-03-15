import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GestureBoard",
  description: "Hand-tracked whiteboard rebuilt with Next.js, MediaPipe, and OpenCV.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
