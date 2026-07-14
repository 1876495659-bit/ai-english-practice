"""Comprehensive Agent Test Script"""
import asyncio
import sys
sys.path.insert(0, '.')

print('='*60)
print('Test 1: Module Import Check')
print('='*60)
try:
    from agents.state import EnglishTutorState
    print('[OK] agents.state imported')
except Exception as e:
    print(f'[FAIL] agents.state: {e}')

try:
    from agents.scenarios import get_scenario_config, list_available_scenarios
    scenarios = list_available_scenarios()
    print(f'[OK] agents.scenarios — scenarios: {scenarios}')
except Exception as e:
    print(f'[FAIL] agents.scenarios: {e}')

try:
    from agents.prompts_loader import load_prompt
    print('[OK] agents.prompts_loader imported')
except Exception as e:
    print(f'[FAIL] agents.prompts_loader: {e}')

try:
    from agents.utils import extract_latest_user_input
    print('[OK] agents.utils imported')
except Exception as e:
    print(f'[FAIL] agents.utils: {e}')

try:
    from config.settings import settings
    print(f'[OK] config.settings — llm_enabled={settings.llm_enabled}, provider={settings.llm_provider}')
except Exception as e:
    print(f'[FAIL] config.settings: {e}')

# Test Scenario Node
print()
print('='*60)
print('Test 2: Scenario Node')
print('='*60)
try:
    from agents.scenario_node import scenario_node
    state = {'scenario': 'interview', 'difficulty': 'medium', 'level': 'intermediate', 'turn': 0, 'messages': [], 'metadata': {}, 'scenario_goal': ''}
    result = asyncio.get_event_loop().run_until_complete(scenario_node(state))
    assert 'messages' in result and len(result['messages']) > 0
    print(f'[OK] scenario_node — opening: {result["messages"][0]["content"][:60]}...')
except Exception as e:
    print(f'[FAIL] scenario_node: {e}')

# Test Conversation Node (mock fallback since llm_enabled=False)
print()
print('='*60)
print('Test 3: Conversation Node (Mock)')
print('='*60)
try:
    from agents.conversation_node import conversation_node
    state = {'scenario': 'daily', 'turn': 1, 'level': 'intermediate', 'difficulty': 'medium', 'metadata': {'scenario_name': '日常对话'}, 'scenario_goal': '提升日常英语交流', 'messages': [{'role': 'assistant', 'content': 'Hello!'}]}
    result = asyncio.get_event_loop().run_until_complete(conversation_node(state))
    assert 'ai_reply' in result and len(result['ai_reply']) > 0
    print(f'[OK] conversation_node — reply: {result["ai_reply"][:60]}...')
except Exception as e:
    print(f'[FAIL] conversation_node: {e}')

# Test Correction Node (rule engine)
print()
print('='*60)
print('Test 4: Correction Node (Rule Engine)')
print('='*60)
try:
    from agents.correction_node import correction_node
    state = {'user_input': 'i go park yesterday', 'scenario': 'daily', 'difficulty': 'medium', 'level': 'intermediate', 'turn': 1, 'messages': []}
    result = asyncio.get_event_loop().run_until_complete(correction_node(state))
    c = result.get('correction', {})
    print(f'  original: {c.get("original", "")}')
    print(f'  has_errors: {c.get("has_errors")}')
    print(f'  errors: {c.get("errors", [])}')
    print(f'  corrected: {c.get("corrected", "")}')
    print(f'  suggestion: {c.get("suggestion", "")}')
    print(f'  polished: {c.get("polished", "")}')
    print(f'[OK] correction_node works')
except Exception as e:
    print(f'[FAIL] correction_node: {e}')

# Test Scoring Node (rule engine)
print()
print('='*60)
print('Test 5: Scoring Node (Rule Engine)')
print('='*60)
try:
    from agents.scoring_node import scoring_node
    state = {'user_input': 'I went to the park yesterday.', 'scenario': 'daily', 'difficulty': 'medium', 'level': 'intermediate', 'turn': 1, 'messages': [], 'retry_count': 0, 'max_retries': 3, 'correction': {'original': 'I went to the park yesterday.'}, 'score': {}, 'skill_progress': {'total_turns': 0, 'avg_score': 0.0, 'error_frequency': {}, 'weakest_dimension': '', 'strongest_dimension': '', 'improvement_trajectory': []}}
    result = asyncio.get_event_loop().run_until_complete(scoring_node(state))
    if hasattr(result, 'goto'):
        print(f'  Command routed to: {result.goto}')
    else:
        s = result.get('score', {})
        scores = s.get('scores', {})
        print(f'  Total score: {s.get("total", 0)}/10')
        print(f'  Scores: fluency={scores.get("fluency")}, grammar={scores.get("grammar")}, vocabulary={scores.get("vocabulary")}, naturalness={scores.get("naturalness")}')
    print(f'[OK] scoring_node works')
except Exception as e:
    print(f'[FAIL] scoring_node: {e}')

# Test Graph Builder + Full Flow
print()
print('='*60)
print('Test 6: Graph Builder + Full LangGraph Flow')
print('='*60)
try:
    from agents.graph_builder import build_graph, reset_checkpointer
    from agents.state import EnglishTutorState

    reset_checkpointer()
    graph = build_graph()
    print(f'[OK] Graph built — type: {type(graph).__name__}')

    initial_state = {
        'scenario': 'restaurant', 'difficulty': 'easy', 'level': 'beginner',
        'scenario_goal': '练习餐厅点餐',
        'ai_reply': '', 'correction': {}, 'score': {}, 'metadata': {},
        'turn': 0, 'retry_count': 0, 'max_retries': 3,
        'session_active': True, 'messages': [],
        'skill_progress': {'total_turns': 0, 'avg_score': 0.0, 'error_frequency': {}, 'weakest_dimension': '', 'strongest_dimension': '', 'improvement_trajectory': []},
    }
    config = {'configurable': {'thread_id': 'flow_test_001'}}
    result = asyncio.get_event_loop().run_until_complete(graph.ainvoke(initial_state, config=config))

    msgs = result.get('messages', [])
    opening = ''
    for m in reversed(msgs):
        c = getattr(m, 'content', None) or (m.get('content') if isinstance(m, dict) else '')
        if c: opening = c; break
    print(f'  Opening: {opening[:60]}...')
    print(f'[OK] Step 1 completed')

    # Step 2: user bad input
    s2 = dict(result)
    s2['turn'] = 1
    s2['messages'] = [{'role': m.type if hasattr(m, 'type') else m['role'], 'content': m.content if hasattr(m, 'content') else m['content']} for m in msgs]
    s2['messages'].append({'role': 'user', 'content': 'i want eat burger'})
    r2 = asyncio.get_event_loop().run_until_complete(graph.ainvoke(s2, config=config))
    corr = r2.get('correction', {})
    errs = corr.get('errors', [])
    print(f'  Bad input errors: {len(errs)}')
    for e in errs: print(f'    - {e}')
    print(f'[OK] Step 2 completed')

    # Step 3: good input
    s3 = dict(r2)
    s3['turn'] = 2
    m3 = r2.get('messages', [])
    s3['messages'] = [{'role': m.type if hasattr(m, 'type') else m['role'], 'content': m.content if hasattr(m, 'content') else m['content']} for m in m3]
    s3['messages'].append({'role': 'user', 'content': 'I would like to order a hamburger with fries and a drink please.'})
    r3 = asyncio.get_event_loop().run_until_complete(graph.ainvoke(s3, config=config))
    sc3 = r3.get('score', {})
    tot3 = sc3.get('total', 0)
    skp = r3.get('skill_progress', {})
    print(f'  Good input total: {tot3}/10')
    print(f'  Skill progress: turns={skp.get("total_turns")}, avg_score={skp.get("avg_score")}')
    print(f'[OK] Step 3 completed')

    print()
    print('='*60)
    print('ALL TESTS PASSED')
    print('='*60)
except Exception as e:
    import traceback
    print(f'[FAIL] Full flow: {e}')
    traceback.print_exc()
