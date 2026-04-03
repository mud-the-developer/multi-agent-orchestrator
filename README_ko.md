# Multi-Agent Orchestration System & Real-Time WebUI

[🇺🇸 English Version](README.md)

이 프로젝트는 다중 에이전트 AI 오케스트레이션 파이프라인(Gemma 4 등 최신 LLM 지원)과 이를 실시간으로 분석하는 **Rust 기반의 고도화된 하드웨어 및 텔레메트리 모니터링 웹 대시보드 (WebUI)** 를 제공합니다.

## 🌟 주요 기능

- **다중 에이전트 플로우 (Multi-Agent Flow)**: 기획자(Planner), 코더(Coder), 비전(Vision) 에이전트 간의 작업을 독립적으로 조율하고 연동합니다.
- **실시간 하드웨어 텔레메트리 (Live Hardware Telemetry)**: `btop`이나 `nvtop` 같은 외부 터미널 툴 없이도, Rust 서버가 직접 커널(`sysinfo`)과 Nvidia 드라이버(`nvml-wrapper`)에서 실시간 자원 사용량을 스캔하여 CPU, System RAM, GPU VRAM 메트릭을 브라우저에 바로 표시합니다.
- **ApexCharts 대시보드**: LLM 토큰 생성 속도 추이(ms 단위)와 하드웨어 사용량을 반응형 애니메이션으로 화려하게 시각화합니다.
- **Mermaid 프로토콜 뷰**: 에이전트들이 서로 대화를 나눌 때, 누가 누구에게 어떤 메시지를 보냈는지를 실시간 브라우저 후킹으로 캡처해 작동 순서 다이어그램(Sequence Diagram)으로 자동으로 그려줍니다.
- **분석 데이터 1-Click CSV 추출**: 웹 UI에 쌓인 텔레메트리 및 추론 통계 지표를 별도의 백엔드 연동 없이 브라우저 내에서 즉시 `.csv`로 다운로드하여 연구용 데이터로 가공할 수 있습니다.

---

## 🚀 실행 구동 가이드

### 1. Rust WebUI 백엔드 구동
웹 UI 서버는 하드웨어 리소를 깊게 모니터링하기 위해 Python 스크립트와 분리된 완전한 독립 포그라운드로 실행됩니다.

새로운 터미널을 열고 다음 명령어를 실행해주세요:
```bash
# WebUI 폴더로 이동
cd webui/

# Rust 서버 빌드 및 실행
cargo run
```
*웹 UI 서버는 `http://0.0.0.0:3123` 포트로 열립니다.*
*(주의: 원활한 GPU 텔레메트리를 위해서는 NVIDIA 드라이버가 구동 중인 리눅스 운영체제를 권장합니다.)*

### 2. 오케스트레이터 프로세스 실행
새로운 터미널 창을 하나 더 열어, AI 연산 및 에이전트 관리를 수행하는 뼈대 스크립트를 실행합니다. 다음과 같이 `--webhook-url` 파라미터를 넘겨주시면 자동으로 웹 대시보드와 통신이 시작됩니다.

```bash
# Strict Vocab (24GB VRAM 한정 세팅) 환경으로 웹소켓 서버를 작성하게 하는 예시:
python multi_agent_hf_gemma4_args_with_gui.py --config gemma4_24gb_strict_vocab.json "Write a tiny websocket chat server" --webhook-url http://0.0.0.0:3123/api/hook
```

### 3. 브라우저 대시보드 접속
두 프로세스(Rust, Python)가 모두 정상적으로 기동되었다면 브라우저에서 다음 주소로 접속해주세요:
[http://127.0.0.1:3123](http://127.0.0.1:3123) (SSH 환경일 경우 접속 중인 외부 IP로 변경)

- **Dashboard & Research 탭**: 가장 먼저 클릭하여, Python의 구동 없이도 로컬 시스템의 CPU와 GPU 점유율이 차트에 제대로 물결치고 있는지 점검하세요.
- **Live Event Stream 탭**: 언어 모델이 추론한 긴 코드나 로그 텍스트를 끊김 없이 스크롤 바를 통해 라이브로 관전할 수 있습니다.
- **Protocol Flow 탭**: 다중 에이전트가 어떤 지시 체계로 통신(Planner -> Coder 방향)하는지 시퀀스 다이어그램으로 확인할 수 있습니다.
