#define PROGRAM_VERSION "Version 1.0 - SIMD"

// ----< Including Files >---------------------------------------------------

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// ----< Configuration >-----------------------------------------------------

// Basic Configuration
#define UINT_T uint64_t
#define PRIUINT PRIu64
#define THREAD_ID      /* DO NOT REMOVE OR EDIT */
#define FILE_SUFFIX    /* DO NOT REMOVE OR EDIT */
#define UNIX_TIMESTAMP /* DO NOT REMOVE OR EDIT */

// Task Configuration
// - note: *** Future Schedules ***
//         Task Configuration will be saved into thread result files!
//         So if you want to continue last progress,
//         you should NOT change Task Configuration!
#define TASK_SEGMENT_SIZE_IN_BYTE (1 * 768 * 1024ULL)
#define COMPUTE_RANGE_START 0ULL
#define COMPUTE_RANGE_LENGTH 1000000000000ULL
#define USING_THREAD_COUNT 8ULL
#define SAVE_THREAD_RESULT_FILENAME "thread_result_" THREAD_ID FILE_SUFFIX
#define SAVE_RESULT_SUMMARY_FILENAME "summary_" UNIX_TIMESTAMP FILE_SUFFIX

// Additional Configuration
#define ENABLE_OUTPUT_SUMMARY_FILE true
#define TABLED_PRIME_UPPER_LIMIT 192ULL
#define WHEELED_PRIME_UPPER_LIMIT 250000ULL
#define USING_WHEEL 30030
#define USING_SIMD true
#define SIMD_TYPE AVX2

// Log Configuration
#define QUIET_MODE false
#define VERBOSE_MODE false

// ----< Macro >-------------------------------------------------------------

#if USING_SIMD == true
#include <immintrin.h>
#endif

#define AVX512F 0x01
#define AVX2 0x02

// MACRO: EXIT CODE
#define EXIT_SUCCESS 0
#define EXIT_FAILURE__MACRO_ASSERTION_FAILED 0x10
#define EXIT_FAILURE__MEMORY_ALLOC_FAILED 0x20
#define EXIT_FAILURE__FILE_OPEN_FAILED 0x30
#define EXIT_FAILURE__CPU_INST_SET_NOT_SUPPORTED_AVX512 0x40
#define EXIT_FAILURE__CPU_INST_SET_NOT_SUPPORTED_AVX2 0x41

#define EXIT_ERROR(error_info, exit_code)                               \
    do                                                                  \
    {                                                                   \
        LOG_ERROR("Error info: %s", error_info);                        \
        LOG_ERROR(                                                      \
            "Fatal Error Occured, Program Exit!\nExit Code = %u.\nSee " \
            "'MACRO: EXIT CODE' for more information.",                 \
            exit_code);                                                 \
        exit(exit_code);                                                \
    } while (0)

#define LOG_LEVEL 2
#define LOG_FUNCTION_NAME_WITH_INHERIT false

#if LOG_LEVEL >= 0

#include <cstdio>

#if LOG_FUNCTION_NAME_WITH_INHERIT == true
#define __LOGGER_BASIC_LOG_FUNCTION_NAME __FUNCTION__
#else
#define __LOGGER_BASIC_LOG_FUNCTION_NAME __func__
#endif

#define __LOGGER_BASIC_LOG_OUT_INFO(fmt, ...)                           \
    printf("[INFO] - %s() " fmt "\n", __LOGGER_BASIC_LOG_FUNCTION_NAME, \
           __VA_ARGS__)
#define __LOGGER_BASIC_LOG_OUT_WARN(fmt, ...)                           \
    printf("[WARN] - %s() " fmt "\n", __LOGGER_BASIC_LOG_FUNCTION_NAME, \
           __VA_ARGS__)
#define __LOGGER_BASIC_LOG_OUT_ERROR(fmt, ...)                           \
    printf("[ERROR] - %s() " fmt "\n", __LOGGER_BASIC_LOG_FUNCTION_NAME, \
           __VA_ARGS__)
#define __LOGGER_BASIC_LOG_EMPTY(fmt, ...)

#if LOG_LEVEL >= 1
#define LOG_ERROR(fmt, ...) __LOGGER_BASIC_LOG_OUT_ERROR(fmt, __VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...) __LOGGER_BASIC_LOG_EMPTY(fmt, __VA_ARGS__)
#endif

#if LOG_LEVEL >= 2
#define LOG_WARN(fmt, ...) __LOGGER_BASIC_LOG_OUT_WARN(fmt, __VA_ARGS__)
#else
#define LOG_WARN(fmt, ...) __LOGGER_BASIC_LOG_EMPTY(fmt, __VA_ARGS__)
#endif

#if LOG_LEVEL >= 3
#define LOG_INFO(fmt, ...) __LOGGER_BASIC_LOG_OUT_INFO(fmt, __VA_ARGS__)
#else
#define LOG_INFO(fmt, ...) __LOGGER_BASIC_LOG_EMPTY(fmt, __VA_ARGS__)
#endif

#endif // LOG_LEVEL >= 0

#define OUTPUT_NOTICE(fmt, ...) printf(fmt, __VA_ARGS__)
#define OUTPUT_TIMER_NOTICE(unit_str, time) \
    printf("Task finished in %Lf " unit_str "s.\n", time)

#if QUIET_MODE == true

#define OUTPUT(fmt, ...)
#define OUTPUT_TIMER(unit_str, time)
#define VERBOSE_OUTPUT(fmt, ...)
#define VERBOSE_OUTPUT_TIMER(unit_str, time)

#undef VERBOSE_MODE
#define VERBOSE_MODE false

#else

#define OUTPUT(fmt, ...) printf(fmt, __VA_ARGS__)
#define OUTPUT_TIMER(unit_str, time) \
    printf("Task finished in %Lf " unit_str "s.\n", time);

#if VERBOSE_MODE == true
#define VERBOSE_OUTPUT(fmt, ...) printf(fmt, __VA_ARGS__)
#define OUTPUT_TIMER_VERBOSE(unit_str, time) \
    printf("Task finished in %Lf " unit_str "s.\n", time)

#else
#define VERBOSE_OUTPUT(fmt, ...)
#define VERBOSE_OUTPUT_TIMER(unit_str, time)
#endif

#endif
// MACRO: GENERATE DATA BY USER CONFIGURATION

#define ASSERTION_STATE 0
#define MACRO_EXCEPTION_INFO ""

#define MACRO_ASSERT                                                        \
    do                                                                      \
    {                                                                       \
        switch (ASSERTION_STATE)                                            \
        {                                                                   \
        case 0:                                                             \
            LOG_INFO("Macro assertion success!%s", "");                     \
            break;                                                          \
        case 1:                                                             \
            LOG_WARN("Macro assertion success, but found Warnings.%s", ""); \
            LOG_WARN("Warning info: %s", MACRO_EXCEPTION_INFO);             \
            break;                                                          \
        case 2:                                                             \
            LOG_ERROR("Macro assertion failed!%s", "");                     \
            EXIT_ERROR(MACRO_EXCEPTION_INFO,                                \
                       EXIT_FAILURE__MACRO_ASSERTION_FAILED);               \
        }                                                                   \
    } while (0)

// Checking Platform

#ifdef _WIN32
#define USER_PLATFORM "Windows"
#else
#ifdef __linux__
#define USER_PLATFORM "Linux"
#else
#undef ASSERTION_STATE
#define ASSERTION_STATE 2
#undef MACRO_EXCEPTION_INFO
#define MACRO_EXCEPTION_INFO "Unsupported Platform!"
#endif
#endif

// Checking __int128_t Support
#ifdef __SIZEOF_INT128__
#define HAS_INT128_T
#elif defined(__GNUC__) && defined(__x86_64__)
#define HAS_INT128_T
#else
// does not support int128
#endif

#ifdef HAS_INT128_T
#undef HAS_INT128_T
#else
#undef ASSERTION_STATE
#define ASSERTION_STATE 2
#undef MACRO_EXCEPTION_INFO
#define MACRO_EXCEPTION_INFO "__int128_t is Not Supported!"
#endif

#if USING_SIMD == true
#if SIMD_TYPE == AVX512F
#define _USING_AVX512F
#define SIMD_TYPE_STR "AVX512F"
#elif SIMD_TYPE == AVX2
#define _USING_AVX2
#define SIMD_TYPE_STR "AVX2"
#else
#undef ASSERTION_STATE
#define ASSERTION_STATE 1
#undef MACRO_EXCEPTION_INFO
#define MACRO_EXCEPTION_INFO "Unsupported SIMD type, fall back to normal mode."
#define SIMD_TYPE_STR "None"
#define _USING_FALLBACK
#endif
#else
#define SIMD_TYPE_STR "None"
#define _USING_FALLBACK
#endif

// Platform specialized aligned malloc & free
#ifdef _WIN32
#define aligned_malloc _aligned_malloc
#define aligned_free _aligned_free
#endif
#ifdef __linux__
#define aligned_malloc(size, align) std::aligned_alloc(align, size)
#define aligned_free(addr) free(addr)
#endif

// Bitmap container (chunk)
// uint64_t
#define BITMAP_CHUNK_SIZE_IN_BYTE 8
#define BITMAP_CHUNK_TYPE uint64_t
#define BITMAP_CHUNK_FORMAT PRIu64
#define BITMAP_CHUNK_SIZE_IN_BIT 64
#ifdef _WIN32
#define POP_COUNT(x) __builtin_popcountll(x)
#define CTZ(x) __builtin_ctzll(x)
#else
#ifdef __linux__
#define POP_COUNT(x) __builtin_popcountl(x)
#define CTZ(x) __builtin_ctzl(x)
#endif
#endif
#define BITMAP_CHUNK_TYPE_LG_SIZE_IN_BIT 6
#define BITMAP_CHUNK_TYPE_SIZE_LOWBIT_MASK 0b111111

#include "wheel.h"

// calculate other data with configuration
#define TASK_SEGMENT_LENGTH_IN_NUMBER (TASK_SEGMENT_SIZE_IN_BYTE * 8 * 2)
#define COMPUTE_RANGE_END (COMPUTE_RANGE_START + COMPUTE_RANGE_LENGTH)
#define THREAD_BITMAP_INITALIZE_CHUNK_COUNT \
    ((TASK_SEGMENT_SIZE_IN_BYTE / BITMAP_CHUNK_SIZE_IN_BYTE) + 8)

// calculate total memory usage
#define TOTAL_MEMORY_USAGE_IN_BYTE                                     \
    (USING_THREAD_COUNT *                                              \
         (TASK_SEGMENT_SIZE_IN_BYTE + 5 * BITMAP_CHUNK_SIZE_IN_BYTE) + \
     WHEEL_M * (2) + WHEEL_PHI * (2 + 1 * 8))

// memory limit: 32MiB (to confirm most bitmaps are in cpu cache)
#define TOTAL_MEMORY_USAGE_WARNING_LIMIT (1ULL << 25)

// warning: memory usage too large
#if TOTAL_MEMORY_USAGE_IN_BYTE > TOTAL_MEMORY_USAGE_WARNING_LIMIT
#undef ASSERTION_STATE
#define ASSERTION_STATE 1
#undef MACRO_EXCEPTION_INFO
#define MACRO_EXCEPTION_INFO \
    "Total memory usage may too large to put into L3 Cache."
#endif
#undef TOTAL_MEMORY_USAGE_WARNING_LIMIT

#define TABLED_PRIME_WARNING_LIMIT 128ULL
#define TABLED_PRIME_ERROR_LIMIT 512ULL
// warning: table too large
#if TABLED_PRIME_UPPER_LIMIT > TABLED_PRIME_WARNING_LIMIT
#undef ASSERTION_STATE
#define ASSERTION_STATE 1
#undef MACRO_EXCEPTION_INFO
#define MACRO_EXCEPTION_INFO "Prime Table may too large to put into L2 Cache."
#endif
// error: table too large
#if TABLED_PRIME_UPPER_LIMIT > TABLED_PRIME_ERROR_LIMIT
#undef ASSERTION_STATE
#define ASSERTION_STATE 2
#undef MACRO_EXCEPTION_INFO
#define MACRO_EXCEPTION_INFO "Prime Table too large!"
#endif
#undef TABLED_PRIME_WARNING_LIMIT
#undef TABLED_PRIME_ERROR_LIMIT

#define TIME_FSTR_BUFFER_SIZE 100

// MACRO: OPERATIONS

// bit operations
#define GET_CHUNK_INDEX(i) ((i) >> BITMAP_CHUNK_TYPE_LG_SIZE_IN_BIT)
#define GET_CHUNK_LOWBIT(i) ((i) & BITMAP_CHUNK_TYPE_SIZE_LOWBIT_MASK)

#define BITMAP_GET(bitmap, x) \
    ((bitmap[GET_CHUNK_INDEX(x)] >> (GET_CHUNK_LOWBIT(x))) & (CHUNK)1)
#define BITMAP_SET_TRUE(bitmap, x) \
    bitmap[GET_CHUNK_INDEX(x)] |= ((CHUNK)1 << GET_CHUNK_LOWBIT(x))
#define BITMAP_SET_FALSE(bitmap, x) \
    bitmap[GET_CHUNK_INDEX(x)] &= (~((CHUNK)1 << GET_CHUNK_LOWBIT(x)))
#define LOW_NBIT_MASK(n) (((CHUNK)1 << n) - (CHUNK)1)
#define EXCEPT_NBIT_MASK(n) (~((CHUNK)1 << n))

// math
#define INF 0xFFFFFFFFFFFF0000
#define FLOORD_DIV(x, divisor) ((x) / (divisor))
#define CEILD_DIV(x, divisor) (((x) + (divisor) - 1) / (divisor))
#define ODD_COUNT_BETWEEN(a, b) (FLOORD_DIV(((b) + 1), 2) - FLOORD_DIV((a), 2))

#define INV(b) ((~0ULL / (uint64_t)(b)) + 1ULL)
#define BARRETT_MU(m) ((__uint128_t)1 << 63) / (m)
#define INV(b) ((~0ULL / (uint64_t)(b)) + 1ULL)

#define CEIL_DIV(a, b, inv_b) \
    ((uint64_t)(((__uint128_t)((uint64_t)(a) + (b) - 1) * (inv_b)) >> 64))

#define BARRETT_MOD(x, m)                                                   \
    ({                                                                      \
        uint64_t _x = (uint64_t)(x);                                        \
        uint32_t _r =                                                       \
            (uint32_t)(_x - (((__uint128_t)_x * BARRETT_MU(m)) >> 63) * m); \
        _r - ((m) & -(uint32_t)(_r >= (m)));                                \
    })

#define CEIL_DIV_MOD(a, b, inv_b, m) BARRETT_MOD(CEIL_DIV(a, b, inv_b), m)

// system
#define PAUSE                                         \
    OUTPUT(" > Press Enter To Continue ...\n%s", ""); \
    (!QUIET_MODE) && std::cin.get()
#define PAUSE_NOTICE                                         \
    OUTPUT_NOTICE(" > Press Enter To Continue ...\n%s", ""); \
    std::cin.get()

#define LIKELY(condition) __builtin_expect(condition, 1)
#define UNLIKELY(condition) __builtin_expect(condition, 0)

#define TIMER_START auto timer = std::chrono::high_resolution_clock::now()

#define TIMER_END(unit_scale)                               \
    (std::chrono::duration_cast<std::chrono::microseconds>( \
         std::chrono::high_resolution_clock::now() - timer) \
         .count() /                                         \
     (long double)unit_scale)

// ----< Structure >---------------------------------------------------------

struct BasePrime
{
    UINT_T *list = nullptr;
    uint64_t *inv = nullptr;
    UINT_T count = 0;
    uint32_t **multiple = nullptr;

    UINT_T small_segment_edge = 0;
    UINT_T medium_segment_edge = 0;
};

struct SmallPrimeTable
{
    BITMAP_CHUNK_TYPE **table = nullptr;
    UINT_T prime_count = 0;
    UINT_T *chunk_count = nullptr;
    UINT_T **ctz_projection = nullptr;
};

struct Wheel
{
    uint16_t m = 0;
    uint16_t phi = 0;
    const uint16_t *w;
    const uint8_t (*half_gap_offset)[7];
    const uint8_t *half_gap8;
    const uint16_t *next_idx;
    Wheel(uint16_t _m, uint16_t _phi, const uint16_t *_w,
          const uint8_t (*_half_gap_offset)[7], const uint8_t *_half_gap8,
          const uint16_t *_next_idx)
        : m(_m), phi(_phi), w(_w), half_gap_offset(_half_gap_offset), half_gap8(_half_gap8), next_idx(_next_idx) {};
};

struct alignas(64) SegmentProperties
{
    UINT_T segment_task_begin_num;
    UINT_T segment_task_end_num;
    UINT_T segment_task_length_num;
    UINT_T segment_bit_count;
    UINT_T last_chunk_bit_count;
    UINT_T segment_chunk_count;
};

struct alignas(64) Task
{
    bool valid = false;
    UINT_T start_number = 0;
    UINT_T length_number = 0;
    UINT_T max_base_prime_index = 0;
    SegmentProperties properties;
};

struct SharedThreadContext
{
    // base prime
    const BasePrime &base;

    // wheel
    const Wheel &wheel;

    // small prime table
    const SmallPrimeTable &table;

    char padding[40];

    SharedThreadContext(const BasePrime &b, const Wheel &w,
                        const SmallPrimeTable &t)
        : base(b), wheel(w), table(t) {};
};

struct alignas(64) LocalThreadContext
{
    UINT_T thread_id;
    BITMAP_CHUNK_TYPE *bitmap = nullptr;
    UINT_T segment_base_prime_count;
    UINT_T segment_base_prime_max;
    UINT_T count = 0;
};

struct alignas(64) Result
{
    UINT_T count = 0;
};

struct alignas(64) Extracted256
{
    uint32_t dat[8];
};

std::queue<Task> global_task_queue;

std::mutex task_mutex;

Result global_thread_result[USING_THREAD_COUNT];

Result global_final_result;

unsigned long long global_start_time;
char global_start_time_fstr[TIME_FSTR_BUFFER_SIZE + 1];
long double global_running_time_in_sec;

// ----< Declaration >-------------------------------------------------------

// bsearch
UINT_T find_first_ge(const UINT_T *arr, const UINT_T n, const UINT_T target);

// get_time
void get_time(unsigned long long *time_stamp, char *format_buffer,
              size_t buffer_size);

// --------< Main Thread >----------------

// Display User Configuration
void display_configuration(FILE *fp = stdout);

// Check User Environment
void check_environment();

// Get Estimated Prime Count with a upper bound 'n'
UINT_T prime_count_estimate(UINT_T n);

// Generate Base Primes
void generate_base_primes(BasePrime &base_primes);

void generate_base_primes_multiple_table(BasePrime &base_primes);

// Generate Tabled Primes
void generate_small_prime_table(const BasePrime &base_primes,
                                SmallPrimeTable &table);

// Generate Wheel
Wheel generate_wheel();

void mark_table_bits(const BasePrime &base_primes,
                     BITMAP_CHUNK_TYPE **&small_prime_table,
                     UINT_T &small_prime_count, UINT_T *&table_chunk_count);

void build_ctz_projection(const BasePrime &base, SmallPrimeTable &table);

void flip_table_bits(const UINT_T small_prime_count,
                     const UINT_T *table_chunk_count,
                     BITMAP_CHUNK_TYPE **&small_prime_table);

// to enable 8-times unroll
void copy_table_mirrors(const UINT_T small_prime_count,
                        const UINT_T *table_chunk_count,
                        BITMAP_CHUNK_TYPE **&small_prime_table);

// Generate Segment Properties
void generate_segment_properties(Task &t);

// Generate Tasks
void generate_task(const BasePrime &base, std::queue<Task> &task_queue);

// Request a Task from Global Task List
Task request_task();

// Output Result to Summary File
void output_summary();

// free all allocated memory
void free_memory(BasePrime &base, SmallPrimeTable &table);

// Main Thread Entrance
int run(int argc, char **argv);

// Program Entrance
int main(int argc, char **argv) { return run(argc, argv); }

// --------< Working Thread >----------------

void working_thread(const UINT_T thread_id, const SharedThreadContext &ctx);

UINT_T process_task(const Task &task, const SharedThreadContext &ctx,
                    LocalThreadContext &loc);

inline void
process_small_prime(const SegmentProperties &properties, const UINT_T prime,
                    const BITMAP_CHUNK_TYPE *table, const UINT_T table_length,
                    const UINT_T *projection, LocalThreadContext &loc);

inline void process_medium_prime(const SegmentProperties &properties,
                                 const UINT_T prime, const UINT_T prime_inv,
                                 const Wheel &wheel, LocalThreadContext &loc);

inline void process_big_prime(const SegmentProperties &properties,
                              const UINT_T prime, const uint32_t *multiples,
                              LocalThreadContext &loc);

// ----< Implementation >----------------------------------------------------

int run(int argc, char **argv)
{
    // Counter Start
    OUTPUT("Start Twin Prime Counter ... %s\n", "");

    // Perform Macro Settings Assertion
    MACRO_ASSERT;

    check_environment();

    // Get Current Time
    get_time(&global_start_time, global_start_time_fstr, TIME_FSTR_BUFFER_SIZE);

    // Display User Settings
    display_configuration();

    // Wait for User Confirmation
    PAUSE;

    // Create Context Data
    BasePrime base;

    SmallPrimeTable table;

    // Preprocess: Compute Base Primes
    generate_base_primes(base);

    generate_base_primes_multiple_table(base);

    // Preprocess: Compute Prime Table
    generate_small_prime_table(base, table);

    // Preprocess: Generate Wheel
    Wheel wheel = generate_wheel();

    // Generate Tasks for Working Thread
    generate_task(base, global_task_queue);

    OUTPUT("Creating Context and Threads ...%s\n", "");

    SharedThreadContext context(const_cast<const BasePrime &>(base),
                                const_cast<const Wheel &>(wheel),
                                const_cast<const SmallPrimeTable &>(table));

    std::thread *threads[USING_THREAD_COUNT];

    OUTPUT("Done!\n\n%s", "");

    PAUSE;

    TIMER_START;

    for (UINT_T i = 0; i < USING_THREAD_COUNT; i++)
    {
        threads[i] = new std::thread(working_thread, i, context);
    }

    for (auto &t : threads)
    {
        t->join();
    }

    global_final_result.count = 0;

    for (UINT_T i = 0; i < USING_THREAD_COUNT; i++)
    {
        global_final_result.count += global_thread_result[i].count;
    }

    global_running_time_in_sec = TIMER_END(1000000ULL);
    OUTPUT_TIMER_NOTICE("second", global_running_time_in_sec);

    for (auto &t : threads)
    {
        delete t;
    }

    free_memory(base, table);

    // Show Result
    OUTPUT_NOTICE("\nCompute Finished! Result = %" PRIUINT ".\n",
                  global_final_result.count);

// Output Summary File if Enabled
#if ENABLE_OUTPUT_SUMMARY_FILE == true
    output_summary();
#endif

    PAUSE_NOTICE;

    // Counter End
    OUTPUT("\nEnd Twin Prime Counter ... %s\n", "");

    // Exit
    return EXIT_SUCCESS;
}

void working_thread(const UINT_T thread_id, const SharedThreadContext &ctx)
{
    OUTPUT("Thread %" PRIUINT " Joined.\n", thread_id);

// define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    Task task;

    LocalThreadContext loc;

    // clear array if not empty
    if (loc.bitmap != nullptr)
    {
        free(loc.bitmap);
        loc.bitmap = nullptr;
    }

    // allocate memory
    loc.bitmap = (CHUNK *)aligned_malloc(
        (THREAD_BITMAP_INITALIZE_CHUNK_COUNT) * sizeof(CHUNK), 64);
    if (!loc.bitmap)
    {
        EXIT_ERROR(
            "Allcate memory for local.bitmap failed! Check your "
            "available memory.",
            EXIT_FAILURE__MEMORY_ALLOC_FAILED);
    }

    // initialize data
    loc.thread_id = thread_id;
    loc.count = 0;

    while ((task = request_task()).valid)
    {
        loc.count += process_task(task, ctx, loc);
    }

    global_thread_result[thread_id].count = loc.count;

    // free
    if (loc.bitmap != nullptr)
    {
        aligned_free(loc.bitmap);
        loc.bitmap = nullptr;
    }

    OUTPUT("Thread %" PRIUINT " Exited.\n", thread_id);

    // undefine typename alias macro
#undef CHUNK
}

UINT_T process_task(const Task &task, const SharedThreadContext &ctx,
                    LocalThreadContext &loc)
{
// define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    UINT_T local_count = 0;

    // --- Segment Properties ---

    const SegmentProperties &p = task.properties;

    const UINT_T LAST_CHUNK_BIT_COUNT =
        p.segment_bit_count & BITMAP_CHUNK_TYPE_SIZE_LOWBIT_MASK;
    const UINT_T LAST_CHUNK_LOWBIT_MASK =
        (LAST_CHUNK_BIT_COUNT == 0 ? ((CHUNK)-1)
                                   : LOW_NBIT_MASK(LAST_CHUNK_BIT_COUNT));

    // --- Initialize Memory ---

    memset(loc.bitmap, 0xFF, (p.segment_chunk_count - 1) * sizeof(CHUNK));
    memset(&loc.bitmap[p.segment_chunk_count - 1], 0x00, sizeof(CHUNK));
    loc.bitmap[p.segment_chunk_count - 2] &= LAST_CHUNK_LOWBIT_MASK;
    if (UNLIKELY(task.start_number == 0))
        loc.bitmap[0] &= EXCEPT_NBIT_MASK(0);

    UINT_T prime_idx = 1;

    // --- Calculate Iteration Edges

    UINT_T small_prime_iteration_end =
        std::min(task.max_base_prime_index, ctx.base.small_segment_edge);

    UINT_T medium_prime_iteration_end =
        std::min(task.max_base_prime_index, ctx.base.medium_segment_edge);

    // --- Process Small Primes ---

    for (; prime_idx < small_prime_iteration_end; prime_idx++)
    {
        process_small_prime(p, ctx.base.list[prime_idx],
                            ctx.table.table[prime_idx],
                            ctx.table.chunk_count[prime_idx],
                            ctx.table.ctz_projection[prime_idx], loc);
    }

    // --- Process Medium Primes ---

    for (; prime_idx < medium_prime_iteration_end; prime_idx++)
    {
        process_medium_prime(p, ctx.base.list[prime_idx],
                             ctx.base.inv[prime_idx], ctx.wheel, loc);
    }

    // --- Process Big Primes ---

    for (; prime_idx < task.max_base_prime_index; prime_idx++)
    {
        process_big_prime(p, ctx.base.list[prime_idx],
                          ctx.base.multiple[prime_idx], loc);
    }

    // --- Restore Original Prime ---

    if (UNLIKELY(p.segment_task_begin_num <=
                 ctx.base.list[ctx.base.count - 1]))
    {
        prime_idx = find_first_ge(ctx.base.list, ctx.base.count,
                                  p.segment_task_begin_num);
        UINT_T prime;
        while ((prime = ctx.base.list[prime_idx]) <= p.segment_task_end_num)
        {
            BITMAP_SET_TRUE(loc.bitmap, prime >> 1);
            prime_idx++;
        }
    }

    // --- Count Twin Primes ---

    for (UINT_T i = 0; i < p.segment_chunk_count; i++)
    {
        local_count +=
            POP_COUNT((loc.bitmap[i]) &
                      ((loc.bitmap[i] >> 1) |
                       (loc.bitmap[i + 1] << (BITMAP_CHUNK_SIZE_IN_BIT - 1))));
    }

    return local_count;

    // undefine typename alias macro
#undef CHUNK
}

UINT_T find_first_ge(const UINT_T *arr, const UINT_T n, const UINT_T target)
{
    UINT_T left = 0, right = n;
    UINT_T result = n;
    while (left < right)
    {
        int mid = left + (right - left) / 2;
        if (arr[mid] < target)
        {
            left = mid + 1;
        }
        else
        {
            right = mid;
        }
    }
    return left;
}

void get_time(unsigned long long *time_stamp, char *format_buffer,
              size_t buffer_size)
{
    time_t raw_time;
    time(&raw_time);
    *time_stamp = (unsigned long long)raw_time;
    struct tm *time_info = localtime(&raw_time);
    strftime(format_buffer, buffer_size, "%Y/%m/%d %H:%M:%S", time_info);
}

UINT_T prime_count_estimate(UINT_T n)
{
    if (n < 2)
        return 0;
    if (n < 60184)
        return n;
    return (UINT_T)ceil(n / (log(n) - 1.15));
}

void generate_base_primes(BasePrime &base_primes)
{
    // define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    // Generator Start
    OUTPUT("Start Generating Base Primes ...%s\n", "");

    // clear result array if not empty
    if (base_primes.list != nullptr)
    {
        free(base_primes.list);
        base_primes.list = nullptr;
    }

    const UINT_T UPPER_LIMIT = (UINT_T)ceil(sqrt(COMPUTE_RANGE_END));
    UINT_T estimate_size = prime_count_estimate(UPPER_LIMIT);
    UINT_T cnt = 0;
    UINT_T bitmap_chunk_count =
        (UINT_T)ceil((UPPER_LIMIT + 1) / (double)BITMAP_CHUNK_SIZE_IN_BIT);

    OUTPUT("Base Prime Upper Limit: %" PRIUINT "\n", UPPER_LIMIT);
    OUTPUT("Estimate Count (upper limit): %" PRIUINT "\n", estimate_size);
    OUTPUT("Bitmap Chunk Count: %" PRIUINT "\n", bitmap_chunk_count);

    // allocate memory
    CHUNK *not_prime = (CHUNK *)malloc(bitmap_chunk_count * sizeof(CHUNK));
    if (!not_prime)
    {
        EXIT_ERROR(
            "Allcate memory for not_prime failed! Check your available memory.",
            EXIT_FAILURE__MEMORY_ALLOC_FAILED);
    }
    memset(not_prime, 0x00, bitmap_chunk_count * sizeof(CHUNK));
    base_primes.list = (UINT_T *)malloc(estimate_size * sizeof(UINT_T));
    if (!base_primes.list)
    {
        EXIT_ERROR(
            "Allcate memory for base_primes failed! Check your "
            "available memory.",
            EXIT_FAILURE__MEMORY_ALLOC_FAILED);
    }

    // temporary variables
    UINT_T prime, composite;

    // Euler sieve (opposite bitmap)
    for (UINT_T i = 2; i <= UPPER_LIMIT; i++)
    {
        if (!BITMAP_GET(not_prime, i))
            base_primes.list[cnt++] = i;
        for (UINT_T j = 0; j < cnt; j++)
        {
            prime = base_primes.list[j];
            composite = i * prime;
            if (composite > UPPER_LIMIT)
                break;
            BITMAP_SET_TRUE(not_prime, composite);
            if (i % prime == 0)
                break;
        }
    }

    OUTPUT("Computed Base Prime Count: %" PRIUINT "\n", cnt);

    base_primes.list[cnt] = INF;
    base_primes.count = cnt;

    // allocate memory for inv of prime
    if (base_primes.inv)
    {
        free(base_primes.inv);
        base_primes.inv = nullptr;
    }
    base_primes.inv = (UINT_T *)malloc((cnt + 1) * sizeof(uint64_t));
    if (!base_primes.inv)
    {
        EXIT_ERROR(
            "Allcate memory for base_prime_inv failed! Check your "
            "available memory.",
            EXIT_FAILURE__MEMORY_ALLOC_FAILED);
    }
    // calculate inv for div
    for (UINT_T i = 0; i <= cnt; i++)
    {
        base_primes.inv[i] = INV(base_primes.list[i]);
    }
    OUTPUT("Computed Mod_Inv for Base Primes.%s\n", "");

    // calculate edge indices
    base_primes.small_segment_edge =
        find_first_ge(base_primes.list, cnt, TABLED_PRIME_UPPER_LIMIT);
    OUTPUT("Found Small Prime Segment Edge: %" PRIUINT "\n",
           base_primes.small_segment_edge);

    base_primes.medium_segment_edge =
        find_first_ge(base_primes.list, cnt, WHEELED_PRIME_UPPER_LIMIT);
    OUTPUT("Found Medium Prime Segment Edge: %" PRIUINT "\n",
           base_primes.medium_segment_edge);

    // undefine typename alias macro
#undef CHUNK

    // Generator End
    OUTPUT("Done!\n%s\n", "");

    return;
}

void generate_base_primes_multiple_table(BasePrime &base_primes)
{
    OUTPUT("Start Generating Multiples of Base Primes ...%s\n", "");

    // free if not empty
    if (base_primes.multiple != nullptr)
    {
        free(base_primes.multiple);
        base_primes.multiple = nullptr;
    }

    // allocate memory
    base_primes.multiple =
        (uint32_t **)malloc((base_primes.count + 1) * sizeof(uint32_t *));
    if (!base_primes.multiple)
    {
        EXIT_ERROR(
            "Allcate memory for base_prime_multiple failed! Check your "
            "available memory.",
            EXIT_FAILURE__MEMORY_ALLOC_FAILED);
    }

    // initialize array
    memset(base_primes.multiple, 0x00,
           (base_primes.count + 1) * sizeof(uint32_t *));

    OUTPUT("Base Prime Count: %" PRIUINT "\n", base_primes.count);

    for (UINT_T i = 1; i <= base_primes.count; i++)
    {
        // free if not empty
        if (base_primes.multiple[i] != nullptr)
        {
            free(base_primes.multiple[i]);
            base_primes.multiple = nullptr;
        }

        // allocate memory
        base_primes.multiple[i] = (uint32_t *)malloc(7 * sizeof(uint32_t));
        if (!base_primes.multiple[i])
        {
            EXIT_ERROR(
                "Allcate memory for base_prime_multiple[i] failed! "
                "Check your available memory.",
                EXIT_FAILURE__MEMORY_ALLOC_FAILED);
        }

        for (UINT_T j = 0; j < 7; j++)
        {
            base_primes.multiple[i][j] = base_primes.list[i] * (j + 2);
        }
    }

    OUTPUT("Done!\n\n%s", "");
}

void generate_small_prime_table(const BasePrime &base_primes,
                                SmallPrimeTable &table)
{
    // define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    // Generator Start
    OUTPUT("Start Generating Base Primes ...%s\n", "");

    // --- Calculate Count of Small Primes ---

    table.prime_count = base_primes.small_segment_edge;

    OUTPUT("Found %" PRIUINT " small primes.\n", table.prime_count);

    // --- Mark All the bits of Multiple of the Primes ---

    mark_table_bits(base_primes, table.table, table.prime_count,
                    table.chunk_count);

    // --- Calculate 'Count Trailing Zeros' Projection ---

    build_ctz_projection(base_primes, table);

    // --- Flip bits to Make It a '&=' Mask ---

    flip_table_bits(table.prime_count, table.chunk_count, table.table);

    // --- Copy the Beginning of Each Item of the Table to the Tail of Item ---

    copy_table_mirrors(table.prime_count, table.chunk_count, table.table);

// undefine typename alias macro
#undef CHUNK

    // Generator End
    OUTPUT("Done!\n%s\n", "");

    return;
}

Wheel generate_wheel()
{
    OUTPUT("Start Generating Wheel ...%s\n", "");
    Wheel wheel(WHEEL_M, WHEEL_PHI, WHEEL_W, WHEEL_HALF_GAP_OFFSET,
                WHEEL_HALF_GAP8, WHEEL_NEXT_IDX);
    OUTPUT("Using Wheel_%d, Phi: %d\n", WHEEL_M, WHEEL_PHI);
    OUTPUT("Done!\n\n%s", "");
    return wheel;
}

void mark_table_bits(const BasePrime &base_primes,
                     BITMAP_CHUNK_TYPE **&small_prime_table,
                     UINT_T &small_prime_count, UINT_T *&table_chunk_count)
{
    // --- Mark All the bits of Multiple of the Primes ---

    // define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    // clear if not empty
    if (small_prime_table != nullptr)
    {
        free(small_prime_table);
        small_prime_table = nullptr;
    }

    // clear if not empty
    if (table_chunk_count != nullptr)
    {
        free(table_chunk_count);
        table_chunk_count = nullptr;
    }

    // allocate memory
    small_prime_table = (CHUNK **)malloc(small_prime_count * sizeof(UINT_T *));
    if (!small_prime_table)
    {
        EXIT_ERROR(
            "Allcate memory for small_prime_table failed! Check your "
            "available memory.",
            EXIT_FAILURE__MEMORY_ALLOC_FAILED);
    }

    table_chunk_count = (UINT_T *)malloc(small_prime_count * sizeof(UINT_T));
    if (!small_prime_table)
    {
        EXIT_ERROR(
            "Allcate memory for table_chunk_count failed! Check your "
            "available memory.",
            EXIT_FAILURE__MEMORY_ALLOC_FAILED);
    }

    // special process for 2
    table_chunk_count[0] = 0;

    // mark all multiple bits of each small prime to 1 (except 2)
    for (UINT_T i = 1; i < small_prime_count; i++)
    {
        // current prime to process
        UINT_T prime = base_primes.list[i];

        // count of chunks in a table
        table_chunk_count[i] = 0;

        // count of chunks in a group
        UINT_T group_chunk_count = CEILD_DIV(prime, sizeof(CHUNK) * 8);

        // count of groups in a table
        UINT_T group_count = prime;

        // count of chunks in a table
        table_chunk_count[i] = group_chunk_count * group_count;

        // count of bits in a table
        UINT_T total_bit_count =
            table_chunk_count[i] * BITMAP_CHUNK_SIZE_IN_BIT;

        // allocate memory
        small_prime_table[i] = (CHUNK *)aligned_malloc(
            CEILD_DIV((table_chunk_count[i] + 8) * sizeof(CHUNK), 64) * 64, 64);
        if (!small_prime_table[i])
        {
            EXIT_ERROR(
                "Allcate memory for small_prime_table[i] failed! Check your "
                "available memory.",
                EXIT_FAILURE__MEMORY_ALLOC_FAILED);
        }
        memset(small_prime_table[i], 0x00,
               CEILD_DIV((table_chunk_count[i] + 8) * sizeof(CHUNK), 64) * 64);

        // set initial position to the first pos of prime
        // note: bitmap only store odds
        UINT_T initial_pos = FLOORD_DIV(prime, 2) + 1;

        for (UINT_T pos = initial_pos; pos < total_bit_count; pos += prime)
        {
            BITMAP_SET_TRUE(small_prime_table[i], pos);
        }

        VERBOSE_OUTPUT("Prime: %" PRIUINT ", Table Size: %" PRIUINT
                       " Chunks.\n",
                       prime, table_chunk_count[i]);
    }

    OUTPUT("Marked All Multiple bits of Small Primes.%s\n", "");

// undefine typename alias macro
#undef CHUNK
}

void build_ctz_projection(const BasePrime &base, SmallPrimeTable &table)
{
    // --- Calculate 'Count Trailing Zeros' Projection ---

    // define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    // clear if not empty
    if (table.ctz_projection != nullptr)
    {
        free(table.ctz_projection);
        table.ctz_projection = nullptr;
    }

    // allocate memory
    table.ctz_projection =
        (UINT_T **)malloc(table.prime_count * sizeof(UINT_T));
    if (!table.ctz_projection)
    {
        EXIT_ERROR(
            "Allcate memory for table.ctz_projection failed! Check your "
            "available memory.",
            EXIT_FAILURE__MEMORY_ALLOC_FAILED);
    }

    // special process for 2
    table.ctz_projection[0] = nullptr;

    for (int i = 1; i < table.prime_count; i++)
    {
        // allocate memory
        table.ctz_projection[i] = (UINT_T *)aligned_malloc(
            sizeof(UINT_T) * BITMAP_CHUNK_SIZE_IN_BIT, 64);

        if (!table.ctz_projection[i])
        {
            EXIT_ERROR(
                "Allcate memory for small_prime_table[i] failed! Check your "
                "available memory.",
                EXIT_FAILURE__MEMORY_ALLOC_FAILED);
        }

        memset(table.ctz_projection[i], 0,
               sizeof(UINT_T) * BITMAP_CHUNK_SIZE_IN_BIT);

        const CHUNK *cur_table = table.table[i];

        for (UINT_T j = 0; j < table.chunk_count[i]; j++)
        {
            if (cur_table[j])
                table.ctz_projection[i][CTZ(cur_table[j])] = j;
        }

        UINT_T prime;

        for (UINT_T j = (prime = base.list[i]); j < BITMAP_CHUNK_SIZE_IN_BIT;
             j++)
        {
            table.ctz_projection[i][j] = table.ctz_projection[i][j % prime];
        }
    }

    OUTPUT("Built 'CTZ -> Table Index' Projection.%s\n", "");

// undefine typename alias macro
#undef CHUNK
}

void flip_table_bits(const UINT_T small_prime_count,
                     const UINT_T *table_chunk_count,
                     BITMAP_CHUNK_TYPE **&small_prime_table)
{
    // --- Flip bits to Make It a '&=' Mask ---

    for (UINT_T i = 1; i < small_prime_count; i++)
    {
        for (UINT_T j = 0; j < table_chunk_count[i]; j++)
        {
            small_prime_table[i][j] = ~small_prime_table[i][j];
        }
    }

    OUTPUT("Flipped All bits of the Table to Make It a Mask.%s\n", "");
}

void copy_table_mirrors(const UINT_T small_prime_count,
                        const UINT_T *table_chunk_count,
                        BITMAP_CHUNK_TYPE **&small_prime_table)
{
    // --- Copy the Beginning of Each Item of the Table to the Tail of Item ---

// define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    // set unroll times in a loop
    const UINT_T LOOP_UNROLL_TIMES = 8;

    // make a summary of the count of chunks
    UINT_T total_chunk_count = 0;

    for (UINT_T i = 1; i < small_prime_count; i++)
    {
        // get and set length of each copy
        UINT_T cur_chunk_count = table_chunk_count[i];
        UINT_T copy_chunk_count = std::min(cur_chunk_count, LOOP_UNROLL_TIMES);

        total_chunk_count += cur_chunk_count + 8;

        UINT_T pos = 0;

        // copy mask from the beginning of the table
        for (; pos < LOOP_UNROLL_TIMES - copy_chunk_count;
             pos += copy_chunk_count)
        {
            memcpy(&small_prime_table[i][cur_chunk_count + pos],
                   &small_prime_table[i][0], copy_chunk_count * sizeof(CHUNK));
        }

        // copy the last which length less than copy_chunk_count
        memcpy(&small_prime_table[i][cur_chunk_count + pos],
               &small_prime_table[i][pos],
               (LOOP_UNROLL_TIMES - pos) * sizeof(CHUNK));
    }

    OUTPUT("Copied Table Mirrors. Total Chunk Count: %" PRIUINT "\n",
           total_chunk_count);

    // undefine typename alias macro
#undef CHUNK
}

void generate_segment_properties(Task &t)
{
// define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE
    t.properties.segment_task_begin_num =
        t.start_number & 1 ? t.start_number : t.start_number + 1;
    t.properties.segment_task_end_num = t.start_number + t.length_number;
    t.properties.segment_task_length_num = t.length_number;
    t.properties.segment_bit_count = ODD_COUNT_BETWEEN(
        t.properties.segment_task_begin_num, t.properties.segment_task_end_num);
    t.properties.segment_chunk_count =
        CEILD_DIV(t.properties.segment_bit_count,
                  (8 * sizeof(BITMAP_CHUNK_TYPE))) +
        1;

// undefine typename alias macro
#undef CHUNK
}

void generate_task(const BasePrime &base, std::queue<Task> &task_queue)
{
    OUTPUT("Start Generating Thread Tasks ...%s\n", "");

    // pop all the tasks if not empty
    while (!task_queue.empty())
        task_queue.pop();

    const UINT_T length_in_number = TASK_SEGMENT_LENGTH_IN_NUMBER;

    Task temp_task;
    temp_task.valid = true;
    temp_task.start_number = COMPUTE_RANGE_START;
    temp_task.length_number = length_in_number + 1;

    UINT_T task_count = 0;
    UINT_T prime_index = 0;
    UINT_T prime = 0;

#if VERBOSE_MODE == true
    UINT_T task_total_estimate =
        CEILD_DIV(COMPUTE_RANGE_LENGTH, length_in_number);
    UINT_T verbose_output_message_count = 30;
    UINT_T verbose_output_message_interval =
        CEILD_DIV(task_total_estimate, verbose_output_message_count - 1);
#endif

    while (temp_task.start_number + length_in_number < COMPUTE_RANGE_END)
    {
        while (prime_index <= base.count)
        {
            prime = base.list[prime_index];
            if (prime * prime > temp_task.start_number + length_in_number)
                break;
            prime_index++;
        }
#if VERBOSE_MODE == true
        if (task_count % verbose_output_message_interval == 0)
        {
            VERBOSE_OUTPUT("Task %" PRIUINT ": [%" PRIUINT ", %" PRIUINT
                           "], with Max Base Prime: P(%" PRIUINT ")\n",
                           task_count, temp_task.start_number,
                           temp_task.start_number + temp_task.length_number,
                           prime_index);
        }
#endif
        temp_task.max_base_prime_index = prime_index;
        generate_segment_properties(temp_task);
        task_queue.push(temp_task);
        temp_task.start_number += length_in_number;
        task_count++;
    }

    // jump the last task's base primes

    while (prime_index <= base.count)
    {
        prime = base.list[prime_index];
        if (prime * prime > temp_task.start_number + length_in_number)
            break;
        prime_index++;
    }
    temp_task.max_base_prime_index = prime_index;

    temp_task.length_number = COMPUTE_RANGE_END - temp_task.start_number;
    generate_segment_properties(temp_task);
    task_queue.push(temp_task);
    task_count++;
#if VERBOSE_MODE == true
    VERBOSE_OUTPUT("Task %" PRIUINT ": [%" PRIUINT ", %" PRIUINT
                   "], with Max Base Prime: P(%" PRIUINT ")\n",
                   task_count, temp_task.start_number,
                   temp_task.start_number + temp_task.length_number,
                   prime_index);
#endif
    OUTPUT("Generated Task Count: %" PRIUINT "\n", task_count);
    OUTPUT("Last Task Length: %" PRIUINT "\n", temp_task.length_number);
    OUTPUT("Done!\n%s\n", "");
}

Task request_task()
{
    bool is_empty;
    Task temp_task;
#if VERBOSE_MODE == true
    size_t remaining;
#endif

    // check if the task queue empty and get the first task
    {
        std::lock_guard<std::mutex> lock(task_mutex);
        is_empty = global_task_queue.empty();
        if (!is_empty)
        {
            temp_task = global_task_queue.front();
            global_task_queue.pop();
#if VERBOSE_MODE == true
            remaining = global_task_queue.size();
#endif
        }
    }

    // tasks cleared, return empty task to notice thread quit
    if (is_empty)
    {
        VERBOSE_OUTPUT("Tasks has been cleared.%s\n", "");
        return Task{false, 0, 0};
    }

    // return the first task in the task queue
    VERBOSE_OUTPUT("Requested Task: %" PRIUINT ", Remaining: %zu\n",
                   temp_task.start_number, remaining);

    return temp_task;
}

void output_summary()
{
    char filename[512];
    sprintf(filename,
            SAVE_RESULT_SUMMARY_FILENAME
            "%llu"
            ".txt",
            global_start_time);

    // open summary file
    FILE *fp = fopen(filename, "w");
    if (fp == nullptr)
    {
        EXIT_ERROR("Cannot Open Summary File!", EXIT_FAILURE__FILE_OPEN_FAILED);
    }

    fprintf(fp, "Twin Primes Counter: %s.\n\n", PROGRAM_VERSION);
    fprintf(fp, "Start Time: %s.\n\n", global_start_time_fstr);
    display_configuration(fp);
    fprintf(fp, "\n > Result: %" PRIUINT ".\n", global_final_result.count);
    fprintf(fp, " > Time Used: %Lf seconds.\n", global_running_time_in_sec);

    fclose(fp);

    OUTPUT("Result Saved to File: '%s'.\n\n", filename);
}

void free_memory(BasePrime &base, SmallPrimeTable &table)
{
    if (base.list != nullptr)
    {
        free(base.list);
        base.list = nullptr;
    }

    for (UINT_T i = 1; i <= base.count; i++)
    {
        if (base.multiple[i] != nullptr)
        {
            free(base.multiple[i]);
            base.multiple[i] = nullptr;
        }
    }

    for (UINT_T i = 1; i < table.prime_count; i++)
    {
        if (table.table[i] != nullptr)
        {
            aligned_free(table.table[i]);
            table.table[i] = nullptr;
        }
        if (table.ctz_projection[i] != nullptr)
        {
            aligned_free(table.ctz_projection[i]);
            table.ctz_projection[i] = nullptr;
        }
    }

    if (base.multiple != nullptr)
    {
        free(base.multiple);
        base.multiple = nullptr;
    }

    if (table.table != nullptr)
    {
        free(table.table);
        table.table = nullptr;
    }

    if (table.ctz_projection != nullptr)
    {
        free(table.ctz_projection);
        table.ctz_projection = nullptr;
    }

    if (table.chunk_count != nullptr)
    {
        free(table.chunk_count);
        table.chunk_count = nullptr;
    }
}

void display_configuration(FILE *fp)
{
    fprintf(fp, "\n------------< User Settings >------------\n");
    fprintf(fp, " - COMPUTE -%s\n", "");
    fprintf(fp, "Compute Range: [%lld, %lld].\n", COMPUTE_RANGE_START,
            COMPUTE_RANGE_END);
    fprintf(fp, "Range Length: %lld.\n", COMPUTE_RANGE_LENGTH);
    fprintf(fp, "Tabled Prime Range: [0, %lld).\n", TABLED_PRIME_UPPER_LIMIT);
    fprintf(fp, "Wheeled Prime Range: [%lld, %lld).\n",
            TABLED_PRIME_UPPER_LIMIT, WHEELED_PRIME_UPPER_LIMIT);
    fprintf(fp, "Using Wheel: M = %d, PHI = %d.\n", WHEEL_M, WHEEL_PHI);
    fprintf(fp, "\n - SEGMENT -%s\n", "");
    fprintf(fp, "Segment Size: %lld Bytes.\n", TASK_SEGMENT_SIZE_IN_BYTE);
    fprintf(fp, "Segment Length: %lld Numbers\n",
            TASK_SEGMENT_LENGTH_IN_NUMBER);
    fprintf(fp, "Bitmap Chunk Size: %d Bytes, or %d bits.\n",
            BITMAP_CHUNK_SIZE_IN_BYTE, BITMAP_CHUNK_SIZE_IN_BIT);
    fprintf(fp, "Using 'uint%d_t' as Bitmap Chunk Type.\n",
            BITMAP_CHUNK_SIZE_IN_BIT);
    fprintf(fp, "\n - SYSTEM -%s\n", "");
    fprintf(fp, "Platform: %s.\n", USER_PLATFORM);
    fprintf(fp, "SIMD: %s.\n", SIMD_TYPE_STR);
    fprintf(fp, "Using Thread Count: %llu.\n", USING_THREAD_COUNT);
    fprintf(fp, "Total Memory Usage: %llu Bytes.\n",
            TOTAL_MEMORY_USAGE_IN_BYTE);
    fprintf(fp, "\n------------------------------------------\n\n");
}

void check_environment()
{
#ifdef _USING_AVX512F
    if (!__builtin_cpu_supports("avx512f"))
    {
        EXIT_ERROR("CPU does Not support AVX512F!",
                   EXIT_FAILURE__CPU_INST_SET_NOT_SUPPORTED_AVX512);
    }
#endif
#ifdef _USING_AVX2
    if (!__builtin_cpu_supports("avx2"))
    {
        EXIT_ERROR("CPU does Not support AVX2!",
                   EXIT_FAILURE__CPU_INST_SET_NOT_SUPPORTED_AVX2);
    }
#endif
#ifdef _USING_FALLBACK
#endif
}

inline void process_medium_prime(const SegmentProperties &properties,
                                 const UINT_T prime, const UINT_T prime_inv,
                                 const Wheel &wheel, LocalThreadContext &loc)
{
    // define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE
    // start_pos: start position on bitmap
    UINT_T start_pos =
        CEIL_DIV(std::max(properties.segment_task_begin_num, prime * prime),
                 prime, prime_inv);
    // OUTPUT("%d Start_pos: %" PRIUINT "\n", loc.thread_id, start_pos);
    UINT_T rest = BARRETT_MOD(start_pos, WHEEL_M);
    // UINT_T rest = (start_pos % WHEEL_M);
    start_pos -= rest;
    // OUTPUT("%d rest: %" PRIUINT "\n", loc.thread_id, rest);
    rest = wheel.next_idx[rest];
    // OUTPUT("%d rest: %" PRIUINT "\n", loc.thread_id, rest);
    UINT_T wheel_idx = rest;
    start_pos += wheel.w[rest];
    start_pos = ((start_pos * prime) - properties.segment_task_begin_num) >> 1;

    if (UNLIKELY(properties.segment_bit_count < start_pos))
        return;

    UINT_T k_max =
        CEIL_DIV(properties.segment_bit_count - start_pos, prime, prime_inv);

    // OUTPUT("%d Final start_pos: %" PRIUINT "\n", loc.thread_id, start_pos);
    // k0 - k7: multiple to prime, to mark (start_pos + k_n * prime)
    Extracted256 k, kr;
    k.dat[0] = (0), k.dat[1] = (wheel.half_gap_offset[wheel_idx][0]),
    k.dat[2] = (wheel.half_gap_offset[wheel_idx][1]),
    k.dat[3] = (wheel.half_gap_offset[wheel_idx][2]),
    k.dat[4] = (wheel.half_gap_offset[wheel_idx][3]),
    k.dat[5] = (wheel.half_gap_offset[wheel_idx][4]),
    k.dat[6] = (wheel.half_gap_offset[wheel_idx][5]),
    k.dat[7] = (wheel.half_gap_offset[wheel_idx][6]);
    while (k.dat[7] < k_max)
    {
        BITMAP_SET_FALSE(loc.bitmap, k.dat[0] * prime + start_pos);
        BITMAP_SET_FALSE(loc.bitmap, k.dat[1] * prime + start_pos);
        BITMAP_SET_FALSE(loc.bitmap, k.dat[2] * prime + start_pos);
        BITMAP_SET_FALSE(loc.bitmap, k.dat[3] * prime + start_pos);
        BITMAP_SET_FALSE(loc.bitmap, k.dat[4] * prime + start_pos);
        BITMAP_SET_FALSE(loc.bitmap, k.dat[5] * prime + start_pos);
        BITMAP_SET_FALSE(loc.bitmap, k.dat[6] * prime + start_pos);
        BITMAP_SET_FALSE(loc.bitmap, k.dat[7] * prime + start_pos);
        k.dat[0] += wheel.half_gap8[wheel_idx + 0];
        k.dat[1] += wheel.half_gap8[wheel_idx + 1];
        k.dat[2] += wheel.half_gap8[wheel_idx + 2];
        k.dat[3] += wheel.half_gap8[wheel_idx + 3];
        k.dat[4] += wheel.half_gap8[wheel_idx + 4];
        k.dat[5] += wheel.half_gap8[wheel_idx + 5];
        k.dat[6] += wheel.half_gap8[wheel_idx + 6];
        k.dat[7] += wheel.half_gap8[wheel_idx + 7];
        wheel_idx += 8;
        wheel_idx -= ((wheel_idx < WHEEL_PHI) - 1ULL) & WHEEL_PHI;
    }
    for (int i = 0; i < 7; i++)
    {
        if (k.dat[i] < k_max)
        {
            BITMAP_SET_FALSE(loc.bitmap, start_pos + k.dat[i] * prime);
        }
        else
        {
            return;
        }
    }
    // undefine typename alias macro
#undef CHUNK
}

inline void process_big_prime(const SegmentProperties &properties,
                              const UINT_T prime, const uint32_t *multiples,
                              LocalThreadContext &loc)
{
    // define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE
    const UINT_T UNROLL_EXPAND_TIMES = 8;
    const UINT_T prime_m0 = 0;
    const UINT_T prime_m1 = prime;
    const UINT_T prime_m2 = multiples[0];
    const UINT_T prime_m3 = multiples[1];
    const UINT_T prime_m4 = multiples[2];
    const UINT_T prime_m5 = multiples[3];
    const UINT_T prime_m6 = multiples[4];
    const UINT_T prime_m7 = multiples[5];
    const UINT_T prime_m8 = multiples[6];
    UINT_T pos = CEILD_DIV(properties.segment_task_begin_num, prime) * prime;
    pos = std::max(pos, prime * prime);
    pos += ((pos & 1) - 1) & prime;
    pos = ((pos - properties.segment_task_begin_num) >> 1);
    while (pos + prime_m8 < properties.segment_bit_count)
    {
        BITMAP_SET_FALSE(loc.bitmap, pos + prime_m0);
        BITMAP_SET_FALSE(loc.bitmap, pos + prime_m1);
        BITMAP_SET_FALSE(loc.bitmap, pos + prime_m2);
        BITMAP_SET_FALSE(loc.bitmap, pos + prime_m3);
        BITMAP_SET_FALSE(loc.bitmap, pos + prime_m4);
        BITMAP_SET_FALSE(loc.bitmap, pos + prime_m5);
        BITMAP_SET_FALSE(loc.bitmap, pos + prime_m6);
        BITMAP_SET_FALSE(loc.bitmap, pos + prime_m7);
        pos += prime_m8;
    }
    while (pos < properties.segment_bit_count)
    {
        BITMAP_SET_FALSE(loc.bitmap, pos);
        pos += prime;
    }
    // undefine typename alias macro
#undef CHUNK
}

// --------< Thread Function Implementations >-------------------------------

#ifdef _USING_FALLBACK

void process_small_prime(const SegmentProperties &properties,
                         const UINT_T prime, const BITMAP_CHUNK_TYPE *table,
                         const UINT_T table_length, const UINT_T *projection,
                         LocalThreadContext &loc)
{
    // define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    // calculate start pos
    UINT_T pos = CEILD_DIV(properties.segment_task_begin_num, prime) * prime;
    pos = std::max(pos, prime * prime);
    pos += ((pos & 1) - 1) & prime;
    pos = ((pos - properties.segment_task_begin_num) >> 1);

    // the indexes where copy will start
    UINT_T bitmap_idx = GET_CHUNK_INDEX(pos);
    UINT_T copy_idx = projection[pos & BITMAP_CHUNK_TYPE_SIZE_LOWBIT_MASK];

    const UINT_T UNROLL_EXPAND_TIMES = 8;
    // sub mod_compensate value before do the fake mod,
    // to confirm the value always in range [0, prime - 1]
    UINT_T mod_compensate = FLOORD_DIV(UNROLL_EXPAND_TIMES, prime) * prime;

    // temporary index variable for parallel compute
    UINT_T _temp_bi, _temp_ci;
    while (bitmap_idx + UNROLL_EXPAND_TIMES < properties.segment_chunk_count)
    {
        _temp_bi = bitmap_idx;
        _temp_ci = copy_idx;
        __builtin_prefetch(&loc.bitmap[_temp_bi + 8]);
        loc.bitmap[_temp_bi + 0] &= table[_temp_ci + 0];
        loc.bitmap[_temp_bi + 1] &= table[_temp_ci + 1];
        loc.bitmap[_temp_bi + 2] &= table[_temp_ci + 2];
        loc.bitmap[_temp_bi + 3] &= table[_temp_ci + 3];
        loc.bitmap[_temp_bi + 4] &= table[_temp_ci + 4];
        loc.bitmap[_temp_bi + 5] &= table[_temp_ci + 5];
        loc.bitmap[_temp_bi + 6] &= table[_temp_ci + 6];
        loc.bitmap[_temp_bi + 7] &= table[_temp_ci + 7];
        copy_idx += (UNROLL_EXPAND_TIMES - mod_compensate);
        copy_idx -= (((copy_idx < prime) - 1ULL) & prime);
        bitmap_idx += UNROLL_EXPAND_TIMES;
    }
    while (bitmap_idx < properties.segment_chunk_count)
    {
        loc.bitmap[bitmap_idx] &= table[copy_idx];
        copy_idx++;
        copy_idx -= (((copy_idx < prime) - 1ULL) & prime);
        bitmap_idx++;
    }
    return;

    // undefine typename alias macro
#undef CHUNK
}

#endif

#ifdef _USING_AVX2

#pragma GCC target("avx2,fma")

void process_small_prime(const SegmentProperties &properties,
                         const UINT_T prime, const BITMAP_CHUNK_TYPE *table,
                         const UINT_T table_length, const UINT_T *projection,
                         LocalThreadContext &loc)
{
    // define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    // calculate start pos
    UINT_T pos = CEILD_DIV(properties.segment_task_begin_num, prime) * prime;
    pos = std::max(pos, prime * prime);
    pos += ((pos & 1) - 1) & prime;
    pos = ((pos - properties.segment_task_begin_num) >> 1);

    // the indexes where copy will start
    UINT_T bitmap_idx = GET_CHUNK_INDEX(pos);
    UINT_T copy_idx = projection[pos & BITMAP_CHUNK_TYPE_SIZE_LOWBIT_MASK];

    const UINT_T UNROLL_EXPAND_TIMES = 8;
    // sub mod_compensate value before do the fake mod,
    // to confirm the value always in range [0, prime - 1]
    UINT_T mod_compensate = FLOORD_DIV(UNROLL_EXPAND_TIMES, prime) * prime;

    __m256i vec_b1, vec_b2, vec_t1, vec_t2;

    // temporary index variable for parallel compute
    UINT_T _temp_bi, _temp_ci;
    while (bitmap_idx + UNROLL_EXPAND_TIMES < properties.segment_chunk_count)
    {
        _temp_bi = bitmap_idx;
        _temp_ci = copy_idx;
        __builtin_prefetch(&loc.bitmap[_temp_bi + 8], 0, 0);
        vec_b1 = _mm256_loadu_si256((__m256i *)&loc.bitmap[_temp_bi + 0]);
        vec_b2 = _mm256_loadu_si256((__m256i *)&loc.bitmap[_temp_bi + 4]);
        vec_t1 = _mm256_loadu_si256((__m256i *)&table[_temp_ci + 0]);
        vec_t2 = _mm256_loadu_si256((__m256i *)&table[_temp_ci + 4]);
        _mm256_storeu_si256((__m256i *)&loc.bitmap[_temp_bi + 0],
                            _mm256_and_si256(vec_b1, vec_t1));
        _mm256_storeu_si256((__m256i *)&loc.bitmap[_temp_bi + 4],
                            _mm256_and_si256(vec_b2, vec_t2));
        copy_idx += (UNROLL_EXPAND_TIMES - mod_compensate);
        copy_idx -= (((copy_idx < prime) - 1ULL) & prime);
        bitmap_idx += UNROLL_EXPAND_TIMES;
    }
    while (bitmap_idx < properties.segment_chunk_count)
    {
        loc.bitmap[bitmap_idx] &= table[copy_idx];
        copy_idx++;
        copy_idx -= (((copy_idx < prime) - 1ULL) & prime);
        bitmap_idx++;
    }
    return;

    // undefine typename alias macro
#undef CHUNK
}

#endif

#ifdef _USING_AVX512F

#pragma GCC target("avx512f")

void process_small_prime(const SegmentProperties &properties,
                         const UINT_T prime, const BITMAP_CHUNK_TYPE *table,
                         const UINT_T table_length, const UINT_T *projection,
                         LocalThreadContext &loc)
{
    // define typename alias macro
#define CHUNK BITMAP_CHUNK_TYPE

    // calculate start pos
    UINT_T pos = CEILD_DIV(properties.segment_task_begin_num, prime) * prime;
    pos = std::max(pos, prime * prime);
    pos += ((pos & 1) - 1) & prime;
    pos = ((pos - properties.segment_task_begin_num) >> 1);

    // the indexes where copy will start
    UINT_T bitmap_idx = GET_CHUNK_INDEX(pos);
    UINT_T copy_idx = projection[pos & BITMAP_CHUNK_TYPE_SIZE_LOWBIT_MASK];

    const UINT_T UNROLL_EXPAND_TIMES = 8;
    // sub mod_compensate value before do the fake mod,
    // to confirm the value always in range [0, prime - 1]
    UINT_T mod_compensate = FLOORD_DIV(UNROLL_EXPAND_TIMES, prime) * prime;

    __m512i vec_b, vec_t;

    // temporary index variable for parallel compute
    UINT_T _temp_bi, _temp_ci;
    while (bitmap_idx + UNROLL_EXPAND_TIMES < properties.segment_chunk_count)
    {
        _temp_bi = bitmap_idx;
        _temp_ci = copy_idx;
        __builtin_prefetch(&loc.bitmap[_temp_bi + 8], 0, 0);
        vec_b = _mm512_loadu_si512((__m512i *)&loc.bitmap[_temp_bi]);
        vec_t = _mm512_loadu_si512((__m512i *)&table[_temp_ci]);
        _mm512_storeu_si512((__m512i *)&loc.bitmap[_temp_bi],
                            _mm512_and_si512(vec_b, vec_t));
        copy_idx += (UNROLL_EXPAND_TIMES - mod_compensate);
        copy_idx -= (((copy_idx < prime) - 1ULL) & prime);
        bitmap_idx += UNROLL_EXPAND_TIMES;
    }
    while (bitmap_idx < properties.segment_chunk_count)
    {
        loc.bitmap[bitmap_idx] &= table[copy_idx];
        copy_idx++;
        copy_idx -= (((copy_idx < prime) - 1ULL) & prime);
        bitmap_idx++;
    }
    return;

    // undefine typename alias macro
#undef CHUNK
}

#endif