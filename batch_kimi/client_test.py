#!/usr/bin/env python3
"""
SOS音频检测性能基准测试工具
支持并发测试和性能指标统计
"""

import asyncio
import aiohttp
import argparse
import json
import jwt
import base64
import time
import os
import math
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from statistics import mean, median
import sys


@dataclass
class BenchmarkConfig:
    """基准测试配置"""

    base_url: str = "http://localhost:16415"
    api_key: str = "agorabestvip8"
    audio_file: str = "sample.pcm"
    audio_url: Optional[str] = None  # 音频HTTP URL，如果提供则优先使用
    concurrent_users: int = 10
    total_requests: int = 100
    timeout: int = 30
    delay_between_requests: float = 0.0
    test_mode: str = "chunks"  # "chunks" 或 "single"
    chunk_duration_ms: int = 200
    user_id_prefix: str = "benchmark_user"

    @property
    def url(self) -> str:
        """生成完整的API URL"""
        return f"{self.base_url}/v1/chat/completions"


@dataclass
class RequestResult:
    """单个请求结果"""

    success: bool
    latency: float
    status_code: Optional[int]
    error: Optional[str]
    sos_detected: bool = False
    chunk_idx: Optional[int] = None


class SOSBenchmark:
    """SOS音频检测基准测试类"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.audio_data = None
        self.chunk_data_list = []
        self.results: List[RequestResult] = []
        self.start_time = None
        self.end_time = None

    def generate_token(self) -> str:
        """生成JWT token"""
        payload = {"ts": int(time.time())}
        return jwt.encode(payload, self.config.api_key, algorithm="HS256")

    def _get_audio_format_from_filename(self, filename: str) -> str:
        """根据文件扩展名获取音频格式"""
        ext = os.path.splitext(filename)[1].lower()
        
        # 扩展名到 MIME 类型的映射
        format_map = {
            '.pcm': 'pcm',
            '.raw': 'pcm',
            '.wav': 'wav',
            '.mp3': 'mp3',
        }
        
        # 返回对应的格式，默认为 pcm
        audio_format = format_map.get(ext, 'pcm')
        return audio_format

    def load_audio_data(self) -> bool:
        """加载音频数据并准备测试数据"""
        # 如果提供了 audio_url，直接使用 HTTP URL，不需要加载本地文件
        if self.config.audio_url:
            self.audio_data = self.config.audio_url
            print(f"✅ 使用音频 HTTP URL: {self.audio_data}")
            return True
        
        # 否则从本地文件加载
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_file_path = os.path.join(script_dir, self.config.audio_file)

        if not os.path.exists(audio_file_path):
            print(f"❌ 音频文件不存在: {audio_file_path}")
            return False

        # 根据文件扩展名确定音频格式
        audio_format = self._get_audio_format_from_filename(audio_file_path)
        print(f"📄 检测到音频格式: {audio_format} (基于文件扩展名)")

        try:
            if self.config.test_mode == "single":
                # 单次完整音频模式
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()
                # 使用 data URL 格式，根据文件扩展名指定格式
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                self.audio_data = f"data:audio/{audio_format};base64,{audio_b64}"
                print(f"✅ 加载完整音频文件，大小: {len(audio_bytes)} bytes")

            else:
                # 分块模式
                sample_rate = 16000
                samps_per_ms = sample_rate // 1000

                pcm_data = np.fromfile(audio_file_path, dtype=np.int16)
                total_duration_ms = pcm_data.shape[0] / samps_per_ms
                num_chunks = math.ceil(
                    total_duration_ms / self.config.chunk_duration_ms
                )

                print(f"✅ 音频总时长: {total_duration_ms:.2f}ms")
                print(f"✅ 分块数量: {num_chunks}")
                print(f"✅ 每块时长: {self.config.chunk_duration_ms}ms")

                # 准备所有音频块数据
                for chunk_idx in range(num_chunks):
                    end_sample = (
                        (chunk_idx + 1) * self.config.chunk_duration_ms * samps_per_ms
                    )
                    chunk_pcm_data = pcm_data[: int(end_sample)]
                    chunk_bytes = chunk_pcm_data.tobytes()
                    chunk_data_b64 = base64.b64encode(chunk_bytes).decode("utf-8")
                    # 使用 data URL 格式，根据文件扩展名指定格式
                    chunk_data_url = f"data:audio/{audio_format};base64,{chunk_data_b64}"
                    self.chunk_data_list.append(chunk_data_url)

            return True

        except Exception as e:
            print(f"❌ 加载音频文件失败: {e}")
            return False

    async def send_single_request(
        self, session: aiohttp.ClientSession, user_id: str, request_id: int
    ) -> RequestResult:
        """发送单个请求（完整音频）"""
        payload = {
            "model": "agora_sos_models/finetuned_hf_for_inference_8_1000",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请识别电话沟通场景中如下声音片段的话轮转换意图，判断该片段是否包含明确的开始说话信号。请区分以下两种情况：若检测到清晰语音起始或强烈发言意愿（如语句开头、语气转折），应回复<是>；若仅含附和词（如\"嗯\"、\"yeah\"）、非语言声音（如喷嚏、咳嗽、笑声）、噪声或近似静默等非打断性信号，应回复<否>"
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {"url": self.audio_data},
                        }
                    ]
                }
            ],
            "stream": False,
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": 1,
            "repetition_penalty": 1.0,
            "stop_token_ids": [151667],
        }

        headers = {
            "Content-Type": "application/json",
            # "Authorization": f"Bearer {self.generate_token()}",
        }

        start_time = time.perf_counter()

        try:
            async with session.post(
                self.config.url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                latency = time.perf_counter() - start_time
                response_text = await response.text()
                print(f"The response is: {response.status}\n{response_text}")

                if response.status == 200:
                    try:
                        data = json.loads(response_text)
                        sos_detected = False

                        if "choices" in data and len(data["choices"]) > 0:
                            content = (
                                data["choices"][0].get("message", {}).get("content", "")
                            )
                            if content:
                                print(f"The received content is: {content}")
                                try:
                                    sos_result = json.loads(content)
                                    sos_detected = sos_result.get("sos", False)
                                except json.JSONDecodeError:
                                    pass

                        return RequestResult(
                            success=True,
                            latency=max(0, latency),
                            status_code=response.status,
                            error=None,
                            sos_detected=sos_detected,
                        )
                    except json.JSONDecodeError:
                        return RequestResult(
                            success=False,
                            latency=max(0, latency),
                            status_code=response.status,
                            error="Invalid JSON response",
                        )
                else:
                    return RequestResult(
                        success=False,
                        latency=max(0, latency),
                        status_code=response.status,
                        error=f"HTTP {response.status}: {response_text}",
                    )

        except asyncio.TimeoutError:
            latency = time.perf_counter() - start_time
            return RequestResult(
                success=False,
                latency=max(0, latency),
                status_code=None,
                error="Request timeout",
            )
        except Exception as e:
            latency = time.perf_counter() - start_time
            return RequestResult(
                success=False, latency=max(0, latency), status_code=None, error=str(e)
            )

    async def send_chunk_request(
        self,
        session: aiohttp.ClientSession,
        user_id: str,
        request_id: int,
        chunk_idx: int,
    ) -> RequestResult:
        """发送单个音频块请求"""
        if chunk_idx >= len(self.chunk_data_list):
            return RequestResult(
                success=False,
                latency=0.0,
                status_code=None,
                error="Chunk index out of range",
                chunk_idx=chunk_idx,
            )

        payload = {
            "model": "agora_sos_models/finetuned_hf_for_inference_8_1000",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请识别电话沟通场景中如下声音片段的话轮转换意图，判断该片段是否包含明确的开始说话信号。请区分以下两种情况：若检测到清晰语音起始或强烈发言意愿（如语句开头、语气转折），应回复<是>；若仅含附和词（如\"嗯\"、\"yeah\"）、非语言声音（如喷嚏、咳嗽、笑声）、噪声或近似静默等非打断性信号，应回复<否>"
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {"url": self.chunk_data_list[chunk_idx]},
                        }
                    ]
                }
            ],
            "stream": False,
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": 1,
            "repetition_penalty": 1.0,
            "stop_token_ids": [151667],
        }

        headers = {
            "Content-Type": "application/json",
            # "Authorization": f"Bearer {self.generate_token()}",
        }

        start_time = time.perf_counter()

        try:
            async with session.post(
                self.config.url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            ) as response:
                latency = time.perf_counter() - start_time
                response_text = await response.text()
                print(f"The response is: {response.status}\n{response_text}")

                if response.status == 200:
                    try:
                        data = json.loads(response_text)
                        sos_detected = False

                        if "choices" in data and len(data["choices"]) > 0:
                            content = (
                                data["choices"][0].get("message", {}).get("content", "")
                            )
                            if content:
                                print(f"The received content is: {content}")
                                try:
                                    sos_result = json.loads(content)
                                    sos_detected = sos_result.get("sos", False)
                                except json.JSONDecodeError:
                                    pass

                        return RequestResult(
                            success=True,
                            latency=max(0, latency),
                            status_code=response.status,
                            error=None,
                            sos_detected=sos_detected,
                            chunk_idx=chunk_idx,
                        )
                    except json.JSONDecodeError:
                        return RequestResult(
                            success=False,
                            latency=max(0, latency),
                            status_code=response.status,
                            error="Invalid JSON response",
                            chunk_idx=chunk_idx,
                        )
                else:
                    return RequestResult(
                        success=False,
                        latency=max(0, latency),
                        status_code=response.status,
                        error=f"HTTP {response.status}: {response_text}",
                        chunk_idx=chunk_idx,
                    )

        except asyncio.TimeoutError:
            latency = time.perf_counter() - start_time
            return RequestResult(
                success=False,
                latency=max(0, latency),
                status_code=None,
                error="Request timeout",
                chunk_idx=chunk_idx,
            )
        except Exception as e:
            latency = time.perf_counter() - start_time
            return RequestResult(
                success=False,
                latency=max(0, latency),
                status_code=None,
                error=str(e),
                chunk_idx=chunk_idx,
            )

    async def worker(
        self, session: aiohttp.ClientSession, worker_id: int, requests_per_worker: int
    ) -> List[RequestResult]:
        """工作协程，处理分配给该worker的请求"""
        results = []
        user_id = f"{self.config.user_id_prefix}_{worker_id}"

        for request_id in range(requests_per_worker):
            if self.config.test_mode == "single":
                result = await self.send_single_request(session, user_id, request_id)
                results.append(result)
            else:
                # 分块模式：随机选择一个chunk进行测试
                chunk_idx = request_id % len(self.chunk_data_list)
                result = await self.send_chunk_request(
                    session, user_id, request_id, chunk_idx
                )
                results.append(result)

            # 请求间延迟
            if self.config.delay_between_requests > 0:
                await asyncio.sleep(self.config.delay_between_requests)

        return results

    async def run_benchmark(self) -> bool:
        """运行基准测试"""
        print(f"🚀 开始基准测试...")
        print(
            f"📊 配置: {self.config.concurrent_users} 并发用户, {self.config.total_requests} 总请求"
        )
        print(f"🎵 测试模式: {self.config.test_mode}")

        # 加载音频数据
        if not self.load_audio_data():
            return False

        # 计算每个worker的请求数
        requests_per_worker = self.config.total_requests // self.config.concurrent_users
        remaining_requests = self.config.total_requests % self.config.concurrent_users

        self.start_time = time.time()

        # 创建aiohttp会话，设置合适的连接池参数
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users,
            ttl_dns_cache=300,
            use_dns_cache=True,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout, connect=10, sock_read=self.config.timeout
        )
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, connector_owner=True
        ) as session:
            # 创建工作任务
            tasks = []
            for worker_id in range(self.config.concurrent_users):
                worker_requests = requests_per_worker
                if worker_id < remaining_requests:
                    worker_requests += 1

                task = asyncio.create_task(
                    self.worker(session, worker_id, worker_requests)
                )
                tasks.append(task)

            # 等待所有任务完成
            print("⏳ 执行测试中...")
            worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        self.end_time = time.time()

        # 收集所有结果
        for worker_result in worker_results:
            if isinstance(worker_result, Exception):
                print(f"❌ Worker异常: {worker_result}")
                continue
            self.results.extend(worker_result)

        return True

    def print_results(self):
        """打印测试结果统计"""
        if not self.results:
            print("❌ 没有测试结果")
            return

        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests * 100

        # 延迟统计
        successful_latencies = [r.latency for r in self.results if r.success]
        all_latencies = [r.latency for r in self.results]

        # SOS检测统计
        sos_detected_count = sum(
            1 for r in self.results if r.success and r.sos_detected
        )

        # 错误统计
        error_counts = {}
        for result in self.results:
            if not result.success and result.error:
                error_counts[result.error] = error_counts.get(result.error, 0) + 1

        # 状态码统计
        status_counts = {}
        for result in self.results:
            if result.status_code:
                status_counts[result.status_code] = (
                    status_counts.get(result.status_code, 0) + 1
                )

        total_duration = self.end_time - self.start_time
        requests_per_second = total_requests / total_duration

        print("\n" + "=" * 60)
        print("📊 基准测试结果")
        print("=" * 60)
        print(f"总请求数:           {total_requests}")
        print(f"成功请求数:         {successful_requests}")
        print(f"失败请求数:         {failed_requests}")
        print(f"成功率:             {success_rate:.2f}%")
        print(f"测试总时长:         {total_duration:.2f}s")
        print(f"请求速率:           {requests_per_second:.2f} req/s")
        print(f"SOS检测数量:        {sos_detected_count}")

        if successful_latencies:
            print(f"\n⏱️  延迟统计 (成功请求):")
            print(f"平均延迟:           {mean(successful_latencies)*1000:.2f}ms")
            print(f"中位延迟:           {median(successful_latencies)*1000:.2f}ms")
            print(f"最小延迟:           {min(successful_latencies)*1000:.2f}ms")
            print(f"最大延迟:           {max(successful_latencies)*1000:.2f}ms")

            # 计算百分位数
            sorted_latencies = sorted(successful_latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            if p95_idx < len(sorted_latencies):
                print(f"95%延迟:            {sorted_latencies[p95_idx]*1000:.2f}ms")
            if p99_idx < len(sorted_latencies):
                print(f"99%延迟:            {sorted_latencies[p99_idx]*1000:.2f}ms")

        if all_latencies:
            print(f"\n⏱️  延迟统计 (所有请求):")
            print(f"平均延迟:           {mean(all_latencies)*1000:.2f}ms")

        if status_counts:
            print(f"\n📈 HTTP状态码分布:")
            for status, count in sorted(status_counts.items()):
                print(
                    f"HTTP {status}:           {count} ({count/total_requests*100:.1f}%)"
                )

        if error_counts:
            print(f"\n❌ 错误类型分布:")
            for error, count in sorted(
                error_counts.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"{error}:  {count}")

        print("=" * 60)


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description="SOS音频检测性能基准测试工具")

    parser.add_argument(
        "--base-url",
        default="http://localhost:16415",
        help="服务器基础地址 (默认: http://localhost:16415)",
    )
    parser.add_argument(
        "--api-key", default="", help="API密钥 (默认: agorabestvip8)"
    )
    parser.add_argument(
        "--audio-file", default="sample.pcm", help="音频文件路径 (默认: sample.pcm)"
    )
    parser.add_argument(
        "--audio-url", default=None, help="音频HTTP URL (如提供则优先使用，支持http://或file://格式)"
    )
    parser.add_argument(
        "-c", "--concurrent", type=int, default=10, help="并发用户数 (默认: 10)"
    )
    parser.add_argument(
        "-n", "--requests", type=int, default=100, help="总请求数 (默认: 100)"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="请求超时时间（秒） (默认: 30)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="请求间延迟（秒） (默认: 0.0)"
    )
    parser.add_argument(
        "--mode",
        choices=["chunks", "single"],
        default="chunks",
        help="测试模式: chunks=分块模式, single=完整音频模式 (默认: chunks)",
    )
    parser.add_argument(
        "--chunk-duration", type=int, default=200, help="音频块时长（毫秒） (默认: 200)"
    )
    parser.add_argument(
        "--user-prefix",
        default="benchmark_user",
        help="用户ID前缀 (默认: benchmark_user)",
    )

    return parser


async def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    # 创建配置
    config = BenchmarkConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        audio_file=args.audio_file,
        audio_url=args.audio_url,
        concurrent_users=args.concurrent,
        total_requests=args.requests,
        timeout=args.timeout,
        delay_between_requests=args.delay,
        test_mode=args.mode,
        chunk_duration_ms=args.chunk_duration,
        user_id_prefix=args.user_prefix,
    )

    # 创建并运行基准测试
    benchmark = SOSBenchmark(config)

    try:
        success = await benchmark.run_benchmark()
        if success:
            benchmark.print_results()
        else:
            print("❌ 基准测试失败")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        if benchmark.results:
            print("📊 显示已完成的结果:")
            benchmark.print_results()
        sys.exit(0)
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        sys.exit(1)
    finally:
        # 确保所有异步资源都被清理
        try:
            # 等待一小段时间让连接完全关闭
            await asyncio.sleep(0.1)
            print("🧹 清理完成")
        except Exception as e:
            print(f"⚠️ 清理过程中发生异常: {e}")


if __name__ == "__main__":
    asyncio.run(main())
