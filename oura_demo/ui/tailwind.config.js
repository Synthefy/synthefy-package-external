/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // OURA Dark Theme Background
        'oura-bg': {
          dark: '#1a1815',
          darker: '#0f0e0c',
        },
        // OURA Warm Glass Tints
        'oura-glass': {
          warm: 'rgba(140, 125, 105, 0.45)',
          light: 'rgba(160, 145, 125, 0.35)',
          border: 'rgba(180, 165, 145, 0.25)',
        },
        // OURA Sleep Stage Colors (exact from app)
        'oura-sleep': {
          awake: '#f5f5f5',
          rem: '#7eb8da',
          light: '#a8c8dc',
          deep: '#4a7c9b',
        },
        // OURA Accent Colors
        'oura-heart': '#e57373',
        'oura-teal': '#07ad98',
        'oura-amber': '#f5a623',
        // OURA Text Colors
        'oura-text': {
          primary: '#f5f2ed',
          secondary: 'rgba(245, 242, 237, 0.7)',
          muted: 'rgba(245, 242, 237, 0.5)',
        },
      },
      fontFamily: {
        'cormorant': ['"Cormorant Garamond"', 'Georgia', 'serif'],
        'dm-sans': ['"DM Sans"', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '1.25rem',
      },
      boxShadow: {
        'glass': '0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        'glass-hover': '0 12px 48px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
      },
      keyframes: {
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(10px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in-up': {
          from: { opacity: '0', transform: 'translateY(20px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'liquid-shine': {
          '0%': { backgroundPosition: '200% 50%' },
          '100%': { backgroundPosition: '-200% 50%' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.4s ease-out',
        'fade-in-up': 'fade-in-up 0.5s ease-out',
        'shimmer': 'shimmer 1.5s infinite',
        'liquid-shine': 'liquid-shine 8s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
