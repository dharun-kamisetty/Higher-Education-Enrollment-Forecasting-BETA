import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/providers";
import EduCastChat from "@/components/EduCastChat";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Higher Education Enrollment Forecasting",
  description: "AISHE enrollment forecasts for Indian states",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>{children}</Providers>
        <EduCastChat />
      </body>
    </html>
  );
}
