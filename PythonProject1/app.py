import streamlit as st
import time
import json
from llm.client import XiaohongshuWriter, create_writer
from prompts.xiaohongshu_template import (
    get_complete_prompt,
    HOT_KEYWORDS,
    WRITING_STYLES,
    OPENING_METHODS
)

# 页面配置
st.set_page_config(
    page_title="小红书爆款文案生成器",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #FF2E63;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FF2E63, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF2E63, #FF8E53);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 28px;
        border: none;
        transition: all 0.3s;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 46, 99, 0.3);
    }
    .result-box {
        background: linear-gradient(135deg, #FFF5F7 0%, #FFFAFA 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #FF2E63;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .keyword-tag {
        display: inline-block;
        background: #FFE4E9;
        color: #FF2E63;
        padding: 4px 12px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .keyword-tag:hover {
        background: #FF2E63;
        color: white;
        transform: scale(1.05);
    }
    .keyword-tag.selected {
        background: #FF2E63;
        color: white;
        font-weight: bold;
    }
    .section-header {
        color: #FF2E63;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #FFE4E9;
        padding-bottom: 0.5rem;
    }
    .stat-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #FFE4E9;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# 初始化会话状态
def init_session_state():
    """初始化所有会话状态"""
    defaults = {
        "history": [],
        "current_result": "",
        "selected_keywords": ["绝绝子", "建议收藏"],
        "api_connected": False,
        "writer": None,
        "generation_count": 0,
        "user_api_key": "",  # 用户输入的API密钥
        "api_provider": "阿里云百炼 (通义千问)",  # API提供商
        "last_api_test": None,  # 上次API测试结果
        "api_usage_count": 0,  # API使用次数统计
        "model_settings": {  # 模型相关设置
            "temperature": 0.7,
            "max_tokens": 2000,
            "max_length": 500
        },
        # 添加以下三个新的状态
        "selected_topic": "",  # 存储选择的主题
        "topic_updated": False,  # 标记主题是否更新
        "temp_topic": ""  # 临时存储主题
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# 必须调用初始化函数！
init_session_state()

# 在 init_session_state() 函数调用后添加回调函数
# 定义回调函数
def set_topic_callback(topic_text):
    """设置主题的回调函数"""
    st.session_state.selected_topic = topic_text
    st.session_state.topic_updated = True
    st.session_state.temp_topic = topic_text
    # 添加这一行来触发页面重新渲染
    st.rerun()

# 侧边栏 - 配置区域
with st.sidebar:
    st.markdown("### ⚙️ 创作配置")

    # API设置
    with st.expander("🔑 API设置", expanded=True):
        # 显示当前API状态
        api_status = "❌ 未设置" if not st.session_state.get("user_api_key") else "✅ 已设置"
        st.markdown(f"**API状态:** {api_status}")

        # API提供商选择
        api_provider = st.selectbox(
            "选择AI模型",
            ["阿里云百炼 (通义千问)", "DeepSeek", "测试模式"],
            index=0
        )

        # API密钥输入
        api_key = st.text_input(
            "输入API密钥",
            type="password",
            value=st.session_state.get("user_api_key", ""),
            placeholder="在此输入你的API密钥",
            help="必须输入有效的API密钥才能使用生成功能"
        )

        # 保存API密钥到会话状态
        if api_key and api_key != st.session_state.get("user_api_key", ""):
            st.session_state.user_api_key = api_key
            st.success("✅ API密钥已保存到当前会话")

        # API密钥管理按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 保存密钥", use_container_width=True):
                if api_key:
                    st.session_state.user_api_key = api_key
                    st.success("✅ API密钥已保存")
                    st.rerun()
                else:
                    st.warning("请先输入API密钥")

        with col2:
            if st.button("🗑️ 清除密钥", use_container_width=True, type="secondary"):
                if "user_api_key" in st.session_state:
                    del st.session_state.user_api_key
                st.success("✅ API密钥已清除")
                st.rerun()

        # 使用说明 - 使用details标签替代expander
        st.markdown("""
           <details>
               <summary style="cursor: pointer; font-weight: bold; color: #666; margin-top: 15px;">
                   📖 如何获取API密钥
               </summary>
               <div style="padding: 10px; background: #f9f9f9; border-radius: 5px; margin-top: 5px; font-size: 0.9em;">
                   <p><strong>阿里云百炼：</strong></p>
                   <ol>
                       <li>访问 <a href="https://bailian.console.aliyun.com/" target="_blank">https://bailian.console.aliyun.com/</a></li>
                       <li>注册/登录阿里云账号</li>
                       <li>在控制台创建API密钥</li>
                       <li>复制DASHSCOPE_API_KEY</li>
                   </ol>

                   <p><strong>DeepSeek：</strong></p>
                   <ol>
                       <li>访问 <a href="https://platform.deepseek.com/" target="_blank">https://platform.deepseek.com/</a></li>
                       <li>注册/登录DeepSeek账号</li>
                       <li>在API Keys页面创建密钥</li>
                       <li>复制生成的API密钥</li>
                   </ol>

                   <p><em>注意：密钥仅在当前浏览器会话中保存，刷新页面后需重新输入。</em></p>
               </div>
           </details>
           """, unsafe_allow_html=True)

    # 写作风格选择
    with st.expander("🎨 写作风格", expanded=True):
        writing_style = st.selectbox(
            "选择写作风格",
            WRITING_STYLES,
            index=WRITING_STYLES.index("活泼") if "活泼" in WRITING_STYLES else 0
        )

        opening_method = st.selectbox(
            "开篇方法",
            OPENING_METHODS,
            index=OPENING_METHODS.index("提出疑问") if "提出疑问" in OPENING_METHODS else 0
        )

    # 参数调节
    with st.expander("📊 创作参数", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            num_titles = st.slider("标题数量", 3, 8, 5)
        with col2:
            temperature = st.slider("创意度", 0.0, 1.0, 0.7, 0.1)

        max_length = st.slider("正文字数限制", 300, 1000, 500, 50)

    # 爆款关键词选择
    with st.sidebar.expander("🔑 爆款关键词", expanded=True):
        st.caption("选择1-3个关键词（点击选择/取消）")

        # 确保会话状态初始化
        if "selected_keywords" not in st.session_state:
            st.session_state.selected_keywords = ["绝绝子", "建议收藏"]  # 默认选择两个

        # 使用列布局显示关键词
        cols = st.columns(3)
        all_keywords = [
            "好用到哭", "大数据", "教科书般", "小白必看", "宝藏", "绝绝子",
            "神器", "都给我冲", "划重点", "笑不活了", "YYDS", "秘方"
        ]

        # 创建临时的选择状态字典
        keyword_states = {}

        # 第一遍：显示所有checkbox并收集状态
        for idx, keyword in enumerate(all_keywords):
            col_idx = idx % 3
            with cols[col_idx]:
                # 检查当前是否选中
                is_checked = keyword in st.session_state.selected_keywords

                # 使用唯一的key
                checkbox_key = f"kw_checkbox_{keyword}"

                # 显示checkbox
                checked = st.checkbox(
                    keyword,
                    value=is_checked,
                    key=checkbox_key,
                    label_visibility="collapsed"
                )

                # 使用HTML样式显示（更稳定）
                if checked:
                    st.markdown(
                        f'<span style="background:#FF2E63; color:white; padding:4px 8px; border-radius:10px; font-size:0.9em;">{keyword}</span>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<span style="background:#f0f2f6; color:#666; padding:4px 8px; border-radius:10px; font-size:0.9em;">{keyword}</span>',
                        unsafe_allow_html=True)

                keyword_states[keyword] = checked

        # 第二遍：更新会话状态
        selected_count = sum(keyword_states.values())
        if selected_count > 3:
            st.warning("最多选择3个关键词，已自动调整")
            # 只保留前3个选中的
            selected = [k for k, v in keyword_states.items() if v][:3]
            st.session_state.selected_keywords = selected
        else:
            st.session_state.selected_keywords = [k for k, v in keyword_states.items() if v]

        # 显示已选关键词
        if st.session_state.selected_keywords:
            st.markdown("**已选择：**")
            selected_html = " ".join([
                f'<span style="background:#FF2E63; color:white; padding:4px 8px; margin:2px; border-radius:10px; display:inline-block; font-size:0.9em;">{kw}</span>'
                for kw in st.session_state.selected_keywords
            ])
            st.markdown(selected_html, unsafe_allow_html=True)
        else:
            st.info("未选择关键词，将使用默认设置")

    # 统计信息
    st.markdown("### 📈 统计信息")
    st.markdown(f"""
    <div class="stat-box">
    🔢 生成次数: {st.session_state.get('generation_count', 0)}<br>
    🔐 API调用: {st.session_state.get('api_usage_count', 0)} 次<br>
    💾 历史记录: {len(st.session_state.get('history', []))} 条<br>
    🎯 当前关键词: {len(st.session_state.get('selected_keywords', []))} 个<br>
    🔄 API状态: {"✅ 已连接" if st.session_state.get('api_connected', False) else "❌ 未连接"}
    </div>
    """, unsafe_allow_html=True)

    # API连接测试按钮
    if st.session_state.user_api_key:
        if st.button("🔗 测试API连接", use_container_width=True):
            with st.spinner("正在测试API连接..."):
                try:
                    # 根据选择的提供商创建writer
                    api_provider = st.session_state.api_provider
                    if "阿里云" in api_provider or "通义" in api_provider:
                        provider_type = "aliyun"
                    elif "DeepSeek" in api_provider:
                        provider_type = "deepseek"
                    else:
                        provider_type = "aliyun"

                    writer = create_writer(
                        provider=provider_type,
                        user_api_key=st.session_state.user_api_key
                    )
                    success, message = writer.test_connection()

                    if success:
                        st.session_state.api_connected = True
                        st.success(f"✅ 连接成功: {message}")
                    else:
                        st.session_state.api_connected = False
                        st.error(f"❌ 连接失败: {message}")
                except Exception as e:
                    st.session_state.api_connected = False
                    st.error(f"❌ 测试失败: {str(e)}")
    else:
        st.info("🔑 请输入API密钥以测试连接")

# 主页面布局
st.markdown('<h1 class="main-header">📝 小红书爆款文案AI生成器</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">基于大语言模型的智能文案创作工具 | 采用专业的小红书爆款写作技巧</p>',
            unsafe_allow_html=True)

# 创建两列布局
col_input, col_preview = st.columns([2, 1])

with col_input:
    st.markdown('<div class="section-header">💡 输入创作主题</div>', unsafe_allow_html=True)

    # 检查是否有更新的主题
    if st.session_state.topic_updated:
        default_topic = st.session_state.selected_topic
        # 重置更新标记
        st.session_state.topic_updated = False
    else:
        # 如果没有更新，使用临时主题
        default_topic = st.session_state.temp_topic

    # 主题输入区
    topic = st.text_area(
        "请输入文案主题",
        value=default_topic,
        placeholder="例如：周末咖啡厅自习指南\n秋季护肤routine分享\n独居女孩的温馨小窝布置\n新手化妆步骤详解",
        height=140,
        help="描述越具体，生成的文案越精准！",
        key="main_topic_input"
    )

    # 当用户手动输入时，更新临时主题
    if topic != st.session_state.temp_topic:
        st.session_state.temp_topic = topic

    # 热门主题快速选择
    st.markdown('<div class="section-header">🎯 热门主题参考</div>', unsafe_allow_html=True)

    quick_topics = [
        "新手化妆步骤详解", "大学生平价好物分享", "职场通勤穿搭指南",
        "懒人减肥食谱推荐", "租房改造ins风卧室", "周末Brunch探店打卡",
        "健身小白入门指南", "考研复习时间规划", "自媒体博主入门教程"
    ]

    cols = st.columns(3)
    for idx, quick_topic in enumerate(quick_topics):
        with cols[idx % 3]:
            # 使用 on_click 回调函数
            if st.button(
                    quick_topic,
                    key=f"quick_{idx}",
                    use_container_width=True,
                    on_click=set_topic_callback,
                    args=(quick_topic,)
            ):
                pass  # 回调函数会处理

# col_preview 应该在 col_input 外面，与它并列
# 修改生成按钮部分的 has_topic 判断
with col_preview:
    st.markdown('<div class="section-header">✨ 快速开始</div>', unsafe_allow_html=True)

    # API状态提示
    if not st.session_state.get("user_api_key"):
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fff3e0 0%, #ffecb3 100%);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #ff9800;
            margin-bottom: 20px;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 24px; margin-right: 10px;">🔑</span>
                <h4 style="margin: 0; color: #ff5722;">API密钥未设置</h4>
            </div>
            <p style="margin: 5px 0; color: #666;">
                要使用生成功能，请先在左侧边栏的 <strong>"API设置"</strong> 中输入你的API密钥。
            </p>
            <div style="margin-top: 10px;">
                <span style="display: inline-block; background: #ff9800; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.9em; margin-right: 8px;">📋</span>
                <span style="color: #666;">支持阿里云百炼和DeepSeek</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 配置预览
    st.markdown(f"""
    <div class="stat-box">
    📝 <b>当前配置：</b><br>
    🎨 风格: {writing_style}<br>
    📊 标题: {num_titles}个<br>
    🎯 创意度: {temperature}<br>
    🔑 关键词: {len(st.session_state.selected_keywords)}个<br>
    🔐 API状态: {"✅ 已设置" if st.session_state.get("user_api_key") else "❌ 未设置"}
    </div>
    """, unsafe_allow_html=True)

    # 生成按钮
    has_api_key = bool(st.session_state.get("user_api_key"))
    # 使用会话状态中的主题而不是直接使用topic变量
    current_topic = st.session_state.get("temp_topic", "")
    has_topic = bool(current_topic.strip())

    if has_api_key and has_topic:
        button_label = "🚀 开始生成文案"
        button_help = "点击开始生成小红书爆款文案"
    elif not has_api_key:
        button_label = "🔒 需要API密钥"
        button_help = "请先在侧边栏设置API密钥"
    else:
        button_label = "📝 需要主题"
        button_help = "请输入文案主题"

    generate_clicked = st.button(
        button_label,
        type="primary",
        use_container_width=True,
        disabled=not (has_api_key and has_topic),
        help=button_help,
        key="generate_button"
    )

    # 在按钮下方添加提示
    if not has_api_key:
        st.warning("⚠️ 请先在侧边栏设置API密钥")
    elif not has_topic:
        st.info("💡 请输入文案主题")

# 生成文案逻辑
if generate_clicked:
    # 检查是否有主题
    current_topic = st.session_state.get("temp_topic", "").strip()
    if not current_topic:
        st.error("❌ 请输入文案主题！")
        st.stop()

    # 检查API密钥
    if not st.session_state.get("user_api_key"):
        st.error("❌ 请先在侧边栏设置API密钥！")
        st.info("💡 前往左侧边栏的'API设置'，输入你的API密钥以使用生成功能。")
        st.stop()

    # 初始化进度指示器
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 步骤1: 初始化API客户端（使用用户输入的密钥）
        status_text.text("🔄 正在连接AI模型...")
        progress_bar.progress(20)
        time.sleep(0.5)

        try:
            # 根据用户选择的模型初始化
            api_provider = api_provider  # 从下拉框获取的值

            if "阿里云" in api_provider or "通义" in api_provider:
                provider_type = "aliyun"
            elif "DeepSeek" in api_provider:
                provider_type = "deepseek"
            else:
                provider_type = "aliyun"  # 默认

            writer = create_writer(
                provider=provider_type,
                user_api_key=st.session_state.user_api_key
            )

        except ValueError as e:
            st.error(f"❌ API初始化失败: {str(e)}")
            st.stop()

        success, message = writer.test_connection()

        if not success:
            st.error(f"❌ API连接失败: {message}")
            st.info("💡 请检查：\n1. API密钥是否正确\n2. API密钥是否有额度\n3. 网络连接是否正常")
            st.stop()

        # 步骤2: 构建提示词
        status_text.text("📝 正在构建提示词...")
        progress_bar.progress(40)
        time.sleep(0.3)

        # 步骤3: 调用API生成
        status_text.text("🤖 AI正在创作中，请稍候...")
        progress_bar.progress(60)

        # 使用 current_topic 而不是 topic 变量
        result = writer.generate_xiaohongshu(
            subject=current_topic,  # 修改这里
            style=writing_style,
            opening_method=opening_method,
            selected_keywords=st.session_state.selected_keywords,
            num_titles=num_titles,
            temperature=temperature
        )

        # 步骤4: 处理结果
        status_text.text("✨ 正在处理生成结果...")
        progress_bar.progress(80)
        time.sleep(0.2)

        # 保存到会话状态
        st.session_state.current_result = result["content"]
        st.session_state.generation_count += 1
        st.session_state.api_usage_count += 1  # 增加API使用计数
        st.session_state.api_connected = True  # 标记为已连接

        # 添加到历史记录
        history_entry = {
            "topic": current_topic,  # 使用 current_topic
            "style": writing_style,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keywords": st.session_state.selected_keywords,
            "preview": result["content"][:150] + "..." if len(result["content"]) > 150 else result["content"],
            "api_provider": api_provider,  # 记录使用的API提供商
            "api_usage_id": st.session_state.api_usage_count  # 记录API使用ID
        }
        st.session_state.history.append(history_entry)

        # 完成
        progress_bar.progress(100)
        status_text.text("✅ 文案生成完成！")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        st.success("🎉 文案生成成功！")

    except Exception as e:
        st.error(f"❌ 生成过程中出错: {str(e)}")
        st.info("💡 建议：\n1. 检查API密钥是否有效\n2. 确保网络连接正常\n3. 尝试降低创意度或字数限制")

# 显示生成结果
if st.session_state.current_result:
    st.markdown("---")
    st.markdown('<div class="section-header">✨ 生成结果</div>', unsafe_allow_html=True)

    # 结果选项卡
    tab1, tab2, tab3 = st.tabs(["📄 完整文案", "📋 复制代码", "📊 详细信息"])

    with tab1:
        st.markdown(f'<div class="result-box">{st.session_state.current_result}</div>',
                    unsafe_allow_html=True)

    with tab2:
        st.code(st.session_state.current_result, language="markdown", line_numbers=True)

        col_copy, col_download = st.columns(2)
        with col_copy:
            if st.button("📋 复制到剪贴板", use_container_width=True):
                st.toast("已复制到剪贴板！", icon="✅")
        with col_download:
            if st.button("💾 下载为文件", use_container_width=True):
                st.toast("下载功能开发中...", icon="🛠️")

    with tab3:
        st.json({
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic if 'topic' in locals() else "",
            "style": writing_style,
            "keywords": st.session_state.selected_keywords,
            "parameters": {
                "num_titles": num_titles,
                "temperature": temperature,
                "opening_method": opening_method
            },
            "generation_count": st.session_state.generation_count
        })

    # 操作按钮
    st.markdown('<div class="section-header">🛠️ 操作选项</div>', unsafe_allow_html=True)

    col_new, col_save, col_share = st.columns(3)
    with col_new:
        # 使用 emoji 在文本中，而不是 icon 参数
        if st.button("🔄 重新生成", use_container_width=True):
            st.session_state.current_result = ""
            st.rerun()
    with col_save:
        if st.button("⭐ 收藏文案", use_container_width=True):
            st.toast("已添加到收藏夹！", icon="⭐")
    with col_share:
        if st.button("📤 分享结果", use_container_width=True):
            st.toast("分享功能开发中...", icon="🛠️")

# 历史记录区域
if st.session_state.history:
    st.markdown("---")
    with st.expander("📚 生成历史（最近5条）", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{item['topic']}**")
                    st.caption(f"风格: {item['style']} | 时间: {item['time']}")
                    st.write(item['preview'])
                with col2:
                    if st.button("👁️ 查看", key=f"view_{i}", use_container_width=True):
                        # 这里可以实现查看完整历史的功能
                        st.info("查看功能开发中...")

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p>🎯 基于大语言模型的智能文案生成工具 | 采用专业的小红书爆款写作技巧</p>
    <p>✨ <strong>重要：</strong>每个用户需要输入自己的API密钥才能使用生成功能</p>
    <p>🔐 支持阿里云百炼(通义千问)、DeepSeek等大模型API</p>
    <p>📊 当前会话已生成 <strong>{count}</strong> 次，API调用 <strong>{api_count}</strong> 次</p>
    <p>📧 如有问题或建议，请联系开发者</p>
</div>
""".format(
    count=st.session_state.generation_count,
    api_count=st.session_state.api_usage_count
), unsafe_allow_html=True)

# 运行说明
with st.sidebar:
    with st.expander("ℹ️ 使用说明", expanded=False):
        st.markdown("""
        **快速开始：**
        1. 输入文案主题
        2. 调整创作参数
        3. 点击"开始生成"

        **高级技巧：**
        - 使用具体主题描述
        - 尝试不同写作风格
        - 选择合适的关键词
        - 调整创意度控制文案多样性

        **注意事项：**
        - 确保API密钥有效
        - 遵守平台内容规范
        - 生成结果仅供参考
        """)