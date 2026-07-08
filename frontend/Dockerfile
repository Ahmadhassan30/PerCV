# === Stage 1: Build ===
FROM node:20-alpine AS build

WORKDIR /app

# Copy package descriptors from root context
COPY frontend/package.json ./

# Install dependencies (use npm install to dynamically resolve lockfile if missing)
RUN npm install

# Copy configuration and code
COPY frontend/tsconfig.json ./
COPY frontend/tailwind.config.js ./
COPY frontend/postcss.config.js ./
COPY frontend/vite.config.ts ./
COPY frontend/index.html ./
COPY frontend/src/ ./src/

# Accept build argument for Vite environment variable
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# Build production bundle
RUN npm run build

# === Stage 2: Serve ===
FROM nginx:alpine

# Copy built bundle to default Nginx static folder
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
