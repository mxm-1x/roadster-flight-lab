FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# Optional local preview image: static files only.
FROM python:3.11-slim
WORKDIR /app
COPY --from=frontend /build/rocket_landing/web_assets /app
USER 65534
EXPOSE 7860
CMD ["python", "-m", "http.server", "7860", "--bind", "0.0.0.0", "--directory", "/app"]
