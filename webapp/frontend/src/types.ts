export type JobStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface JobSummary {
  id: string;
  kind: "ingest" | "optimize_trigger" | "experiment";
  status: JobStatus;
  created_at: string;
  started_at?: string | null;
  ended_at?: string | null;
  exit_code?: number | null;
  params: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export interface Corpus {
  name: string;
  data_dir: string;
  suggested_persist_dir: string;
  doc_count: number;
  has_index: boolean;
  file_types: string[];
  ingestion_config?: string | null;
}

export interface ArtifactSummary {
  attack_id: string;
  trigger: string;
  target_claim: string;
  poison_doc_id: string;
  harmful_match_phrases: string[];
  encoder_model: string;
  num_adv_passage_tokens: number;
  target_query_ids: string[];
  final_loss?: number | null;
  path: string;
}

export interface ArtifactDetail extends ArtifactSummary {
  poison_doc_text: string;
  loss_history: number[];
  token_ids: number[];
}

export interface QueryEntry {
  query_id: string;
  query: string;
  ground_truth_answer?: string | null;
  category?: string | null;
  has_attack: boolean;
  attack_artifact_path?: string | null;
}

export interface QueryFile {
  path: string;
  queries: QueryEntry[];
}

export interface Defaults {
  ingestion: Record<string, unknown>;
  corpus_cybersec: Record<string, unknown>;
  corpus_bio_papers: Record<string, unknown>;
  trigger_opt: Record<string, unknown>;
  attack_main: Record<string, unknown>;
  attack_poisonedrag: Record<string, unknown>;
  system_orchestrator: Record<string, unknown>;
  system_debate: Record<string, unknown>;
}

export interface SubagentOutput {
  agent_id: string;
  answer: string;
  citations: string[];
  confidence: number;
  rationale: string;
  poison_retrieved: boolean;
  retrieved_doc_ids: string[];
}

export interface OrchestratorOutput {
  final_answer: string;
  selected_evidence: string[];
  winning_subagents: string[];
  final_confidence: number;
  harmful_action_flag: boolean;
  reasoning_summary?: string | null;
}

export interface DebateRound {
  round_num: number;
  stances: Record<string, string>;
  confidences: Record<string, number>;
  messages: Record<string, string>;
}

export interface DebateTranscript {
  rounds: DebateRound[];
  majority_cluster: string[];
  majority_answer: string;
  rounds_used: number;
  stopped_reason: string;
}

export interface RunLog {
  query_id: string;
  attack_condition: string;
  trigger?: string | null;
  ground_truth_answer?: string | null;
  retrieved_doc_ids_per_agent: Record<string, string[]>;
  poison_retrieved: boolean;
  agent_responses: Record<string, SubagentOutput>;
  final_decision?: OrchestratorOutput | null;
  debate_transcript?: DebateTranscript | null;
  metrics: Record<string, number>;
  _logged_at?: string;
}

export interface RunSummary {
  query_id: string;
  attack_condition: string;
  trigger?: string | null;
  poison_retrieved: boolean;
  harmful_action_flag: boolean;
  final_confidence: number | null;
  final_answer: string;
  logged_at?: string;
  has_debate: boolean;
}

export interface IngestRequest {
  data_dir: string;
  persist_dir: string;
  chunk_size: number;
  chunk_overlap: number;
  embed_model: "local" | "openai";
  similarity_top_k: number;
  variant: "auto" | "generic" | "cybersec";
  rebuild: boolean;
}

export interface TriggerOptRequest {
  attack_id: string;
  query_file: string;
  ingestion_config?: string | null;
  target_query_id?: string | null;
  target_claim: string;
  harmful_match_phrases: string[];
  poison_doc_id?: string | null;
  encoder_model: string;
  num_adv_passage_tokens: number;
  num_iter: number;
  num_grad_iter: number;
  num_cand: number;
  per_batch_size: number;
  algo: "ap" | "cpa";
  ppl_filter: boolean;
  n_components: number;
  seed: number;
  device?: string | null;
  max_training_queries: number;
}

export interface ExperimentRequest {
  system: "orchestrator" | "debate" | "single";
  mode: "clean" | "attack";
  threat_model: "targeted" | "global";
  poisoned_subagent_ids: string[];
  attack_id?: string | null;
  query_file: string;
  corpus: string;
  data_dir?: string | null;
  persist_dir?: string | null;
  ingestion_config?: string | null;
  model?: string | null;
  top_k?: number | null;
  num_subagents?: number | null;
  max_rounds?: number | null;
  stable_for?: number | null;
}
