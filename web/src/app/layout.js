import "./globals.css";

export const metadata = {
  title: "OlfacNet Intelligence — Giving Computers the Sense of Smell",
  description: "OlfacNet Intelligence is an AI-powered olfactory platform that predicts odor profiles from molecular structures using deep learning. Analyze 44,000+ molecules across 105 odor categories.",
  keywords: "olfactory AI, odor prediction, molecular analysis, deep learning, GNN, fragrance technology",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
