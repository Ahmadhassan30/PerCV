/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f7ff',
          100: '#e0efff',
          200: '#b8dfff',
          300: '#7cc4ff',
          400: '#39a4ff',
          500: '#0a84ff', // Premium neon blue
          600: '#0066cc',
          700: '#004da3',
          800: '#003370',
          900: '#002047',
        },
        dark: {
          950: '#030712',
          900: '#090d16',
          800: '#111827',
          700: '#1f2937',
          600: '#374151',
        }
      },
      fontFamily: {
        sans: ['Outfit', 'Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
