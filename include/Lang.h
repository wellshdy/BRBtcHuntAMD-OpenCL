#ifndef LANG_H
#define LANG_H

#include <string>
#include <cstdint>

enum class LangId : uint8_t { EN = 0, PT = 1 };

// Global language setting (default EN)
extern LangId g_lang;

inline const char* _(const char* en, const char* pt) {
    return g_lang == LangId::PT ? pt : en;
}

// ── Short labels (progress display) ──
inline const char* LBL_TIME()      { return _("Time",      "Tempo"); }
inline const char* LBL_SPEED()     { return _("Speed",     "Veloc."); }
inline const char* LBL_COUNT()     { return _("Count",     "Total"); }
inline const char* LBL_PROGRESS()  { return _("Progress",  "Prog."); }
inline const char* LBL_KEY()       { return _("Key",       "Chave"); }
inline const char* LBL_CHUNKS()    { return _("Chunks",    "Saltos"); }
inline const char* LBL_VANITY()    { return _("Vanity",    "Vaidade"); }

// ── Config labels ──
inline const char* LBL_SM()            { return _("SM",                  "SM"); }
inline const char* LBL_THREADS_BLOCK() { return _("ThreadsPerBlock",    "Threads/Bloq"); }
inline const char* LBL_BLOCKS()        { return _("Blocks",             "Blocos"); }
inline const char* LBL_TOTAL_THREADS() { return _("Total threads",      "Total threads"); }
inline const char* LBL_BATCH_SIZE()    { return _("Points batch size",  "Lote pontos"); }
inline const char* LBL_BATCHES_SM()    { return _("Batches/SM",         "Lotes/SM"); }
inline const char* LBL_BATCHES_LAUNCH(){ return _("Batches/launch",     "Lotes/lanc."); }
inline const char* LBL_MEM_UTIL()      { return _("Memory utilization", "Uso memoria"); }

// ── Section headers ──
inline const char* HDR_PREPHASE()      { return _("======== PrePhase: GPU Information",    "======== Pre-Fase: Info GPU"); }
inline const char* HDR_PHASE1()        { return _("======== Phase-1: BruteForce",          "======== Fase-1: Forca Bruta"); }
inline const char* HDR_INIT_POINTS()   { return _("Initializing EC points...",             "Inicializando pontos EC..."); }

// ── Results ──
inline const char* RSLT_FOUND()        { return _("======== FOUND MATCH! =================", "======== CHAVE ENCONTRADA! ========="); }
inline const char* RSLT_NOT_FOUND()    { return _("======== KEY NOT FOUND (exhaustive) =====", "======== CHAVE NAO ENCONTRADA (exaustivo)"); }
inline const char* RSLT_PRIVKEY()      { return _("Private Key",        "Chave Privada"); }
inline const char* RSLT_PUBKEY()       { return _("Public Key",         "Chave Publica"); }

// ── Summary ──
inline const char* SUM_HEADER()        { return _("--- Summary ---",           "--- Resumo ---"); }
inline const char* SUM_KEYS()          { return _("Total keys",               "Total chaves"); }
inline const char* SUM_TIME()          { return _("Time",                     "Tempo"); }
inline const char* SUM_AVG_SPEED()     { return _("Avg speed",               "Veloc. med."); }
inline const char* SUM_VANITY()        { return _("Vanity hits",             "Vaidades"); }

// ── Vanity ──
inline const char* VANITY_SAVED()       { return _("Saved",                  "Salvas"); }
inline const char* VANITY_RESULTS_FILE(){ return _("vanity_results.txt",     "vanity_results.txt"); }
inline const char* VANITY_MATCHING()    { return _("Vanity: matching",       "Vaidade: comparando"); }
inline const char* VANITY_HEX_CHARS()   { return _("hex chars of hash160 against target", "caracteres hex do hash160 c/ alvo"); }

// ── Errors ──
inline const char* ERR_RANGE_LARGE()       { return _("Error: range too large.",                "Erro: intervalo muito grande."); }
inline const char* ERR_RANGE_FORMAT()      { return _("Error: range format must be start:end",  "Erro: formato deve ser inicio:fim"); }
inline const char* ERR_INVALID_RANGE()     { return _("Error: invalid range hex",               "Erro: hex invalido no intervalo"); }
inline const char* ERR_INVALID_ADDR()      { return _("Error: invalid P2PKH address",           "Erro: endereco P2PKH invalido"); }
inline const char* ERR_INVALID_HASH160()   { return _("Error: invalid target hash160 hex",      "Erro: hash160 alvo invalido"); }
inline const char* ERR_NO_GPU()            { return _("No compatible GPUs found.",              "Nenhuma GPU compatível encontrada."); }
inline const char* ERR_GRID_FMT()          { return _("Error: --grid expects \"A,B\" (positive integers).", "Erro: --grid espera \"A,B\" (inteiros positivos)."); }
inline const char* ERR_SLICES()            { return _("Error: --slices must be in",             "Erro: --slices deve estar entre"); }
inline const char* ERR_VANITY()            { return _("Error: --vanity expects an integer 1-40 (number of matching hex chars).", "Erro: --vanity espera inteiro 1-40 (qtd caracteres hex)."); }
inline const char* ERR_BATCH_EVEN()        { return _("Error: batch size must be at least 2 and even.", "Erro: tamanho do lote deve ser >= 2 e par."); }
inline const char* ERR_BATCH_LIMIT()       { return _("Error: batch size too large (constant memory limit).", "Erro: lote muito grande (limite memoria constante)."); }
inline const char* ERR_GPU_INDEX()         { return _("Error: invalid GPU index",               "Erro: indice GPU invalido"); }
inline const char* ERR_GPU_FORMAT()        { return _("Error: --gpus must be 'all' or a comma-separated list.", "Erro: --gpus deve ser 'all' ou lista separada por virgula."); }
inline const char* ERR_BOTH_TARGET()       { return _("Error: provide either --address or --target-hash160, not both.", "Erro: informe --address ou --target-hash160, nao ambos."); }

#endif // LANG_H
