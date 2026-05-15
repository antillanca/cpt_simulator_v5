FROM alpine:3.19

RUN apk add --no-cache lua5.4 lua5.4-cjson

# Copy sandbox runner
COPY lua/sandbox_runner.lua /sandbox/sandbox_runner.lua

# Read-only filesystem, no network (enforced at docker run)
ENTRYPOINT ["lua5.4", "/sandbox/sandbox_runner.lua"]
