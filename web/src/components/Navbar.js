"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  const links = [
    { href: "/", label: "Home" },
    { href: "/predict", label: "Predict" },
    { href: "/explore", label: "Explore" },
    { href: "/research", label: "Research" },
  ];

  return (
    <nav className="navbar" id="main-nav">
      <div className="navbar-inner">
        <Link href="/" className="navbar-logo">
          <span className="logo-icon">🧬</span>
          OlfacNet
        </Link>

        <ul className="navbar-links">
          {links.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                className={pathname === link.href ? "active" : ""}
              >
                {link.label}
              </Link>
            </li>
          ))}
          <li>
            <Link href="/predict" className="navbar-cta">
              Try Prediction →
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}
