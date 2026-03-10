#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/../minerl/MCP-Reborn
# </dev/null 避免 patch 在 hunk 失败时等待交互导致卡住
patch -s -p 1 -i ${DIR}/mcp_patch.diff < /dev/null
# Copy cursors over
cp -r ${DIR}/cursors ./src/main/resources
# Ensure all scripts are runnable
chmod +x *
