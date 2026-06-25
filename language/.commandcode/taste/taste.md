# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# debugging
- Verify before suggesting deletion of cached/built artifacts — don't assume corruption without confirming the actual cause of the error. Confidence: 0.65

# modal
- For Modal-based ML workflows in this project, use `modal run` only (not `modal deploy`) — the existing setup runs from cache with `modal run`. Confidence: 0.65
- Use `modal run --detach` (non-blocking) instead of blocking `modal run` with tail/timeout — user wants to run detached and have progress monitored separately. Confidence: 0.75

