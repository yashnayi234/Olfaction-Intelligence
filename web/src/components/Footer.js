import Link from "next/link";

export default function Footer() {
  return (
    <footer className="footer" id="footer">
      <div className="container">
        <div className="footer-grid">
          <div className="footer-brand">
            <h3>🧬 OlfacNet Intelligence</h3>
            <p>
              Pioneering olfactory AI — giving computers the sense of smell
              through deep learning on molecular structures.
            </p>
          </div>

          <div className="footer-column">
            <h4>Platform</h4>
            <Link href="/predict">Predict Odors</Link>
            <Link href="/explore">Explore Dataset</Link>
            <Link href="/research">Research</Link>
          </div>

          <div className="footer-column">
            <h4>Technology</h4>
            <a href="#">Graph Neural Networks</a>
            <a href="#">Morgan Fingerprints</a>
            <a href="#">Attention Fusion</a>
            <a href="#">API Documentation</a>
          </div>

          <div className="footer-column">
            <h4>Connect</h4>
            <a href="https://github.com/yashnayi234/Olfaction-Intelligence" target="_blank" rel="noopener">
              GitHub
            </a>
            <a href="#">Research Papers</a>
            <a href="#">Contact</a>
          </div>
        </div>

        <div className="footer-bottom">
          <span>© {new Date().getFullYear()} OlfacNet Intelligence. All rights reserved.</span>
          <span>Built with PyTorch, Next.js & 🧪</span>
        </div>
      </div>
    </footer>
  );
}
