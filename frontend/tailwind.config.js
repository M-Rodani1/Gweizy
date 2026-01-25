/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./*.{js,ts,jsx,tsx}",
  ],
  theme: {
    screens: {
      'xs': '475px',
      'sm': '640px',
      'md': '768px',
      'lg': '1024px',
      'xl': '1280px',
      '2xl': '1536px',
    },
    extend: {
      spacing: {
        xs: '0.25rem',
        sm: '0.5rem',
        md: '1rem',
        lg: '1.5rem',
        xl: '2rem',
        '2xl': '3rem',
        '3xl': '4rem'
      },
      fontFamily: {
        sans: ['Manrope', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        display: ['Space Grotesk', 'Manrope', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif']
      },
      fontSize: {
        xs: ['0.75rem', { lineHeight: '1.4' }],
        sm: ['0.875rem', { lineHeight: '1.5' }],
        base: ['1rem', { lineHeight: '1.6' }],
        lg: ['1.125rem', { lineHeight: '1.6' }],
        xl: ['1.25rem', { lineHeight: '1.35' }],
        '2xl': ['1.5rem', { lineHeight: '1.25' }],
        '3xl': ['1.875rem', { lineHeight: '1.2' }],
        '4xl': ['2.25rem', { lineHeight: '1.15' }],
        '5xl': ['3rem', { lineHeight: '1.05' }]
      },
      colors: {
        // Standardized surface colors
        surface: {
          DEFAULT: '#0a0e1a',
          secondary: '#111827',
          tertiary: '#1f2937',
        },
        // Standardized border colors
        border: {
          DEFAULT: '#374151',
          muted: '#1f2937',
          accent: '#06b6d4',
        },
        // Accent colors
        accent: {
          DEFAULT: '#06b6d4',
          hover: '#22d3ee',
          muted: 'rgba(6, 182, 212, 0.2)',
        },
      },
      borderRadius: {
        'card': '0.75rem',    // 12px - for cards/widgets
        'button': '0.5rem',   // 8px - for buttons
        'badge': '9999px',    // full - for badges/pills
      },
      boxShadow: {
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.2)',
        'card-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.2)',
        'glow-cyan': '0 0 20px rgba(6, 182, 212, 0.3)',
        'glow-purple': '0 0 20px rgba(139, 92, 246, 0.3)',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'data-change': 'dataChange 0.4s ease'
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        dataChange: {
          '0%': { filter: 'brightness(1.2)', transform: 'scale(1.01)' },
          '100%': { filter: 'brightness(1)', transform: 'scale(1)' }
        },
      },
    },
  },
  plugins: [],
}
