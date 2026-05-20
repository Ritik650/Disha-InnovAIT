export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0E1116",
        soil: "#1A1F26",
        clay: "#262B33",
        wheat: "#E8DAB1",
        leaf: "#65B36B",
        leafdim: "#3B6F3F",
        rust: "#D87B4A",
        warn: "#E2B447",
        muted: "#8A93A0",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
};
