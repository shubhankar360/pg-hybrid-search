# API image. The embedding model is downloaded and smoke-tested at build
# time, so a running container never depends on reaching Hugging Face.
FROM node:22-slim
WORKDIR /app
RUN chown node:node /app
USER node
COPY --chown=node:node package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force
COPY --chown=node:node src ./src
RUN npx tsx src/fetch-model.ts
ENV NODE_ENV=production PORT=8080
EXPOSE 8080
CMD ["npx", "tsx", "src/server.ts"]
