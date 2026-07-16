"""Configuration loading and validation.

Single loader for config/config.yaml, used by every pipeline stage.
Validates presence and types of required keys at startup and fails loudly
on missing configuration, per Repository Architecture v1.0 Section 5.

Not yet implemented — scaffolded in Commit 1; implemented when the first
stage that consumes configuration is built.
"""
