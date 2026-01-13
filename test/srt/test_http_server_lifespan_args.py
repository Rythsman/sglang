"""
python3 -m unittest test_http_server_lifespan_args.TestHttpServerLifespanArgs
"""

import asyncio
import types
import unittest

from fastapi import FastAPI

from sglang.srt.entrypoints import http_server


class TestHttpServerLifespanArgs(unittest.TestCase):
    def test_single_tokenizer_mode_missing_warmup_thread_args_defaults(self):
        app = FastAPI()
        app.is_single_tokenizer_mode = True
        server_args = types.SimpleNamespace()
        app.server_args = server_args

        resolved_server_args, warmup_thread_args, thread_label = asyncio.run(
            http_server._resolve_lifespan_init_args(app)
        )

        self.assertIs(resolved_server_args, server_args)
        self.assertEqual(warmup_thread_args, (server_args, None, None))
        self.assertEqual(thread_label, "Tokenizer")

    def test_single_tokenizer_mode_preserves_warmup_thread_args(self):
        app = FastAPI()
        app.is_single_tokenizer_mode = True
        server_args = types.SimpleNamespace()
        app.server_args = server_args
        app.warmup_thread_args = (server_args, "pipe", "callback")

        resolved_server_args, warmup_thread_args, thread_label = asyncio.run(
            http_server._resolve_lifespan_init_args(app)
        )

        self.assertIs(resolved_server_args, server_args)
        self.assertEqual(warmup_thread_args, (server_args, "pipe", "callback"))
        self.assertEqual(thread_label, "Tokenizer")

    def test_single_tokenizer_mode_missing_server_args_raises(self):
        app = FastAPI()
        app.is_single_tokenizer_mode = True

        with self.assertRaises(RuntimeError):
            asyncio.run(http_server._resolve_lifespan_init_args(app))

