"""
RAGAS Evaluation Script
Evaluates the Manufacturing Quality RAG + Text-to-SQL system using RAGAS metrics.
Also structurally validates quality agent (5-Why / fishbone / CAPA / 8D) outputs.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.rag_service import RAGService
from app.services.sql_service import TextToSQLService
from app.services.router_service import QueryRouter


class RAGEvaluator:
    """Evaluates RAG system using RAGAS metrics and agent structural validation."""

    def __init__(self):
        self.test_queries_path = Path("tests/test_queries.json")
        self.results_path = Path("evaluation_results.json")
        self.embedding_service = None
        self.vector_service = None
        self.rag_service = None
        self.sql_service = None
        self.quality_workflows = None
        self._initialize_services()

    def _initialize_services(self):
        try:
            if settings.OPENAI_API_KEY and settings.PINECONE_API_KEY:
                print("Initializing RAG services...")
                self.embedding_service = EmbeddingService()
                self.vector_service = VectorService()
                self.vector_service.connect_to_index()
                self.rag_service = RAGService()
                print("✓ RAG services initialized")
            else:
                print("WARNING: OpenAI/Pinecone API keys not configured. Document RAG evaluation skipped.")
        except Exception as e:
            print(f"WARNING: Failed to initialize RAG services: {e}")

        try:
            if settings.DATABASE_URL and settings.OPENAI_API_KEY:
                print("Initializing SQL service...")
                self.sql_service = TextToSQLService()
                self.sql_service.complete_training()
                print("✓ SQL service initialized and trained")
            else:
                print("WARNING: DATABASE_URL not configured. SQL evaluation skipped.")
        except Exception as e:
            print(f"WARNING: Failed to initialize SQL service: {e}")

        try:
            if settings.OPENAI_API_KEY:
                from app.services.quality_langgraph import QualityLangGraphWorkflows
                self.quality_workflows = QualityLangGraphWorkflows(self.rag_service, self.sql_service)
                print("✓ Quality agent workflows initialized")
        except Exception as e:
            print(f"WARNING: Failed to initialize quality workflows: {e}")

    def load_test_queries(self) -> List[Dict[str, Any]]:
        if not self.test_queries_path.exists():
            raise FileNotFoundError(f"Test queries file not found: {self.test_queries_path}")
        with open(self.test_queries_path, 'r') as f:
            data = json.load(f)
        return data['test_queries']

    async def run_query(self, test_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test query through the appropriate pipeline.
        AGENT-type queries return structural validation, not RAGAS-scored text.
        """
        query_type = test_query['type']
        question = test_query['question']
        ground_truth = test_query.get('ground_truth', '')

        result = {
            "query_id": test_query['id'],
            "question": question,
            "ground_truth": ground_truth,
            "query_type": query_type,
            "answer": None,
            "contexts": [],
            "error": None,
        }

        try:
            if query_type == "SQL":
                if not self.sql_service:
                    result['error'] = "SQL service not initialized"
                    result['answer'] = "N/A - SQL service unavailable"
                    return result

                sql_result = await self.sql_service.generate_sql_for_approval(question)
                execution_result = await self.sql_service.execute_approved_query(
                    sql_result['query_id'], approved=True
                )
                answer_parts = [
                    f"SQL Query: {execution_result['sql']}",
                    f"Results: {json.dumps(execution_result['results'][:5])}",
                    f"Total rows: {execution_result['result_count']}",
                ]
                result['answer'] = "\n".join(answer_parts)
                result['contexts'] = [execution_result['sql']]

            elif query_type == "DOCUMENTS":
                if not self.rag_service:
                    result['error'] = "RAG service not initialized"
                    result['answer'] = "N/A - RAG service unavailable"
                    return result

                rag_result = await self.rag_service.generate_answer(
                    question=question, top_k=3, namespace="default", include_sources=True
                )
                result['answer'] = rag_result['answer']
                result['contexts'] = [chunk['text'] for chunk in rag_result.get('sources', [])]

            elif query_type == "HYBRID":
                if not self.sql_service or not self.rag_service:
                    result['error'] = "Both SQL and RAG services required for HYBRID"
                    result['answer'] = "N/A - Services unavailable"
                    return result

                sql_result = await self.sql_service.generate_sql_for_approval(question)
                execution_result = await self.sql_service.execute_approved_query(
                    sql_result['query_id'], approved=True
                )
                rag_result = await self.rag_service.generate_answer(
                    question=question, top_k=3, namespace="default", include_sources=True
                )
                answer_parts = [
                    f"SQL Results: {json.dumps(execution_result['results'][:5])}",
                    f"Context from Documents: {rag_result['answer']}",
                ]
                result['answer'] = "\n".join(answer_parts)
                result['contexts'] = [
                    execution_result['sql'],
                    *[chunk['text'] for chunk in rag_result.get('sources', [])],
                ]

            elif query_type == "AGENT":
                result = await self._run_agent_query(test_query, result)

        except Exception as e:
            result['error'] = str(e)
            result['answer'] = f"Error: {str(e)}"

        return result

    async def _run_agent_query(self, test_query: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Run a quality agent workflow and structurally validate the output."""
        if not self.quality_workflows:
            result['error'] = "Quality workflows not initialized"
            result['answer'] = "N/A - Workflows unavailable"
            return result

        question = test_query['question']
        expected_keys = test_query.get('expected_keys', [])
        workflow = test_query.get('workflow', 'five_why')

        try:
            if workflow == 'fishbone':
                output = await self.quality_workflows.run_fishbone(question)
                validation = _validate_fishbone(output, expected_keys)
            elif workflow == '8d':
                output = await self.quality_workflows.run_draft("8d", question, None)
                validation = _validate_8d(output, expected_keys)
            elif workflow == 'capa':
                output = await self.quality_workflows.run_draft("capa", question, None)
                validation = _validate_capa(output, expected_keys)
            else:
                output = await self.quality_workflows.run_five_why(question)
                validation = _validate_five_why(output, expected_keys)

            result['answer'] = json.dumps(output, indent=2)
            result['contexts'] = []
            result['agent_validation'] = validation
            result['agent_passed'] = validation['passed']
        except Exception as e:
            result['error'] = str(e)
            result['answer'] = f"Agent error: {str(e)}"

        return result

    async def run_all_queries(self) -> List[Dict[str, Any]]:
        test_queries = self.load_test_queries()
        results = []
        print(f"\nRunning {len(test_queries)} test queries...")
        print("=" * 60)
        for i, test_query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] {test_query['id']} ({test_query['type']})")
            print(f"Question: {test_query['question']}")
            result = await self.run_query(test_query)
            results.append(result)
            if result['error']:
                print(f"ERROR: {result['error']}")
            elif result['query_type'] == 'AGENT':
                passed = result.get('agent_passed', False)
                print(f"{'✓' if passed else '✗'} Agent structural validation: {'PASS' if passed else 'FAIL'}")
            else:
                print("✓ Query completed")
        print("\n" + "=" * 60)
        return results

    def evaluate_with_ragas(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run RAGAS on DOCUMENTS/SQL/HYBRID results; report agent validation separately."""
        ragas_results = [
            r for r in results
            if not r['error']
            and r['query_type'] in ('DOCUMENTS', 'SQL', 'HYBRID')
            and r['answer'] not in (None, "N/A - Services unavailable")
        ]
        agent_results = [r for r in results if r['query_type'] == 'AGENT' and not r['error']]

        scores: Dict[str, Any] = {
            "evaluated_queries": 0,
            "skipped_queries": len(results) - len(ragas_results) - len(agent_results),
        }

        # RAGAS for retrieval-based queries
        if ragas_results:
            print(f"\nEvaluating {len(ragas_results)} retrieval queries with RAGAS...")
            dataset = Dataset.from_dict({
                "question": [r['question'] for r in ragas_results],
                "answer": [r['answer'] for r in ragas_results],
                "contexts": [r['contexts'] for r in ragas_results],
                "ground_truth": [r['ground_truth'] for r in ragas_results],
            })
            try:
                eval_result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
                scores.update({
                    "faithfulness": float(eval_result['faithfulness']),
                    "answer_relevancy": float(eval_result['answer_relevancy']),
                    "evaluated_queries": len(ragas_results),
                })
                print("\n" + "=" * 60)
                print("RAGAS Evaluation Results:")
                print(f"Faithfulness:      {scores['faithfulness']:.4f} (target: > 0.75)")
                print(f"Answer Relevancy:  {scores['answer_relevancy']:.4f} (target: > 0.80)")
                print("=" * 60)
            except Exception as e:
                print(f"\nERROR during RAGAS evaluation: {e}")
                scores['ragas_error'] = str(e)

        # Structural validation for agent queries
        if agent_results:
            passed = sum(1 for r in agent_results if r.get('agent_passed', False))
            total = len(agent_results)
            scores['agent_validation'] = {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": round(passed / total, 3),
                "details": [
                    {
                        "id": r['query_id'],
                        "workflow": r.get('agent_validation', {}).get('workflow', ''),
                        "passed": r.get('agent_passed', False),
                        "missing_keys": r.get('agent_validation', {}).get('missing_keys', []),
                    }
                    for r in agent_results
                ],
            }
            print(f"\nAgent Validation: {passed}/{total} passed")

        return scores

    def save_results(self, results: List[Dict[str, Any]], scores: Dict[str, Any]):
        output = {
            "evaluation_date": datetime.utcnow().isoformat(),
            "total_queries": len(results),
            "ragas_scores": scores,
            "query_results": results,
            "summary": {
                "faithfulness_target": 0.75,
                "answer_relevancy_target": 0.80,
                "faithfulness_met": scores.get('faithfulness', 0) > 0.75,
                "answer_relevancy_met": scores.get('answer_relevancy', 0) > 0.80,
            },
        }
        with open(self.results_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to: {self.results_path}")

    async def run_evaluation(self):
        print("=" * 60)
        print("Starting Quality Engineering RAG Evaluation")
        print("=" * 60)
        results = await self.run_all_queries()
        scores = self.evaluate_with_ragas(results)
        self.save_results(results, scores)
        print("\n✓ Evaluation complete!")
        return scores


# ── Agent structural validators ────────────────────────────────────────────────

def _check_keys(output: dict, required: list, workflow: str) -> dict:
    missing = [k for k in required if k not in output or not output[k]]
    return {"workflow": workflow, "passed": len(missing) == 0, "missing_keys": missing}


def _validate_five_why(output: dict, expected_keys: list) -> dict:
    required = expected_keys or ["problem_statement", "whys", "root_cause", "recommended_action", "confidence_score"]
    result = _check_keys(output, required, "five_why")
    whys = output.get("whys", [])
    if not isinstance(whys, list) or len(whys) < 3:
        result['passed'] = False
        result['missing_keys'].append("whys (min 3 levels required)")
    score = output.get("confidence_score", -1)
    if not (0.0 <= float(score) <= 1.0):
        result['passed'] = False
        result['missing_keys'].append("confidence_score out of [0,1] range")
    return result


def _validate_fishbone(output: dict, expected_keys: list) -> dict:
    required_bones = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]
    result = _check_keys(output, ["effect", "bones"], "fishbone")
    bones = output.get("bones", {})
    missing_bones = [b for b in required_bones if b not in bones or not bones[b]]
    if missing_bones:
        result['passed'] = False
        result['missing_keys'].extend([f"bone:{b}" for b in missing_bones])
    return result


def _validate_8d(output: dict, expected_keys: list) -> dict:
    required = expected_keys or [
        "problem_statement", "d1_team", "d2_problem_desc", "d3_containment",
        "d4_root_cause", "d5_perm_action", "d6_implemented", "d7_prevention", "d8_closure",
    ]
    return _check_keys(output, required, "8d")


def _validate_capa(output: dict, expected_keys: list) -> dict:
    required = expected_keys or [
        "problem_statement", "root_cause", "corrective_action", "preventive_action", "status",
    ]
    return _check_keys(output, required, "capa")


async def main():
    evaluator = RAGEvaluator()
    await evaluator.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
