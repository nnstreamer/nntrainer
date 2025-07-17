/* <copyright>

    Copyright (C) 1985 Intel Corporation.

    This software and the related documents are Intel copyrighted materials,
    and your use of them is governed by the express license under which they
    were provided to you ("License"). Unless the License provides otherwise,
    you may not use, modify, copy, publish, distribute, disclose or transmit
    this software or the related documents without Intel's prior written permission.

    This software and the related documents are provided as is, with no express or
    implied warranties, other than those that are expressly stated in the License.

</copyright> */

#ifndef __OMP_H
#   define __OMP_H

#   include <stddef.h>
#   include <stdlib.h>
#   include <stdint.h>

#   define KMP_VERSION_MAJOR    5
#   define KMP_VERSION_MINOR    0
#   define KMP_VERSION_BUILD    20250422
#   define KMP_BUILD_DATE       "2025-04-25 19-16-16 UTC"

#   ifdef __cplusplus
    extern "C" {
#   endif

#   define omp_set_affinity_format   ompc_set_affinity_format
#   define omp_get_affinity_format   ompc_get_affinity_format
#   define omp_display_affinity      ompc_display_affinity
#   define omp_capture_affinity      ompc_capture_affinity

#   if defined(_WIN32)
#       define __KAI_KMPC_CONVENTION __cdecl
#       ifndef __KMP_IMP
#           define __KMP_IMP __declspec(dllimport)
#       endif
#   else
#       define __KAI_KMPC_CONVENTION
#       ifndef __KMP_IMP
#           define __KMP_IMP
#       endif
#   endif

    /* schedule kind constants */
    typedef enum omp_sched_t {
        omp_sched_static  = 1,
        omp_sched_dynamic = 2,
        omp_sched_guided  = 3,
        omp_sched_auto    = 4,
        omp_sched_monotonic = 0x80000000
    } omp_sched_t;

    enum {
      omp_unassigned_thread = -2
    };

    /* set API functions */
    extern void   __KAI_KMPC_CONVENTION  omp_set_num_threads (int);
    extern void   __KAI_KMPC_CONVENTION  omp_set_dynamic     (int);
    extern void   __KAI_KMPC_CONVENTION  omp_set_nested      (int);
    extern void   __KAI_KMPC_CONVENTION  omp_set_max_active_levels (int);
    extern void   __KAI_KMPC_CONVENTION  omp_set_schedule          (omp_sched_t, int);

    /* query API functions */
    extern int    __KAI_KMPC_CONVENTION  omp_get_num_threads  (void);
    extern int    __KAI_KMPC_CONVENTION  omp_get_dynamic      (void);
    extern int    __KAI_KMPC_CONVENTION  omp_get_nested       (void);
    extern int    __KAI_KMPC_CONVENTION  omp_get_max_threads  (void);
    extern int    __KAI_KMPC_CONVENTION  omp_get_thread_num   (void);
    extern int    __KAI_KMPC_CONVENTION  omp_get_num_procs    (void);
    extern int    __KAI_KMPC_CONVENTION  omp_in_parallel      (void);
    extern int    __KAI_KMPC_CONVENTION  omp_in_final         (void);
    extern int    __KAI_KMPC_CONVENTION  omp_get_active_level        (void);
    extern int    __KAI_KMPC_CONVENTION  omp_get_level               (void);
    extern int    __KAI_KMPC_CONVENTION  omp_get_ancestor_thread_num (int);
    extern int    __KAI_KMPC_CONVENTION  omp_get_team_size           (int);
    extern int    __KAI_KMPC_CONVENTION  omp_get_thread_limit        (void);
    extern int    __KAI_KMPC_CONVENTION  omp_get_max_active_levels   (void);
    extern void   __KAI_KMPC_CONVENTION  omp_get_schedule            (omp_sched_t *, int *);
    extern int    __KAI_KMPC_CONVENTION  omp_get_max_task_priority   (void);

    /* lock API functions */
    typedef struct omp_lock_t {
        void * _lk;
    } omp_lock_t;

    extern void   __KAI_KMPC_CONVENTION  omp_init_lock    (omp_lock_t *);
    extern void   __KAI_KMPC_CONVENTION  omp_set_lock     (omp_lock_t *);
    extern void   __KAI_KMPC_CONVENTION  omp_unset_lock   (omp_lock_t *);
    extern void   __KAI_KMPC_CONVENTION  omp_destroy_lock (omp_lock_t *);
    extern int    __KAI_KMPC_CONVENTION  omp_test_lock    (omp_lock_t *);

    /* nested lock API functions */
    typedef struct omp_nest_lock_t {
        void * _lk;
    } omp_nest_lock_t;

    extern void   __KAI_KMPC_CONVENTION  omp_init_nest_lock    (omp_nest_lock_t *);
    extern void   __KAI_KMPC_CONVENTION  omp_set_nest_lock     (omp_nest_lock_t *);
    extern void   __KAI_KMPC_CONVENTION  omp_unset_nest_lock   (omp_nest_lock_t *);
    extern void   __KAI_KMPC_CONVENTION  omp_destroy_nest_lock (omp_nest_lock_t *);
    extern int    __KAI_KMPC_CONVENTION  omp_test_nest_lock    (omp_nest_lock_t *);

    /* OpenMP 5.0  Synchronization hints*/
    typedef enum omp_sync_hint_t {
        omp_sync_hint_none           = 0,
        omp_lock_hint_none           = omp_sync_hint_none,
        omp_sync_hint_uncontended    = 1,
        omp_lock_hint_uncontended    = omp_sync_hint_uncontended,
        omp_sync_hint_contended      = (1<<1),
        omp_lock_hint_contended      = omp_sync_hint_contended,
        omp_sync_hint_nonspeculative = (1<<2),
        omp_lock_hint_nonspeculative = omp_sync_hint_nonspeculative,
        omp_sync_hint_speculative    = (1<<3),
        omp_lock_hint_speculative    = omp_sync_hint_speculative,
        kmp_lock_hint_hle            = (1<<16),
        kmp_lock_hint_rtm            = (1<<17),
        kmp_lock_hint_adaptive       = (1<<18)
    } omp_sync_hint_t;

    /* lock hint type for dynamic user lock */
    typedef omp_sync_hint_t omp_lock_hint_t;

    /* hinted lock initializers */
    extern void __KAI_KMPC_CONVENTION omp_init_lock_with_hint(omp_lock_t *, omp_lock_hint_t);
    extern void __KAI_KMPC_CONVENTION omp_init_nest_lock_with_hint(omp_nest_lock_t *, omp_lock_hint_t);

    /* time API functions */
    extern double __KAI_KMPC_CONVENTION  omp_get_wtime (void);
    extern double __KAI_KMPC_CONVENTION  omp_get_wtick (void);

    /* OpenMP 4.0 */
    extern int  __KAI_KMPC_CONVENTION  omp_get_default_device (void);
    extern void __KAI_KMPC_CONVENTION  omp_set_default_device (int);
    extern int  __KAI_KMPC_CONVENTION  omp_is_initial_device (void);
    extern int  __KAI_KMPC_CONVENTION  omp_get_num_devices (void);
    extern int  __KAI_KMPC_CONVENTION  omp_get_num_teams (void);
    extern int  __KAI_KMPC_CONVENTION  omp_get_team_num (void);
    extern int  __KAI_KMPC_CONVENTION  omp_get_cancellation (void);

    /* OpenMP 4.5 */
    extern int   __KAI_KMPC_CONVENTION  omp_get_initial_device (void);
    extern void* __KAI_KMPC_CONVENTION  omp_target_alloc(size_t, int);
    extern void  __KAI_KMPC_CONVENTION  omp_target_free(void *, int);
    extern int   __KAI_KMPC_CONVENTION  omp_target_is_present(const void *, int);
    extern int   __KAI_KMPC_CONVENTION  omp_target_memcpy(void *, const void *, size_t, size_t, size_t, int, int);
    extern int   __KAI_KMPC_CONVENTION  omp_target_memcpy_rect(void *, const void *, size_t, int, const size_t *,
                                            const size_t *, const size_t *, const size_t *, const size_t *, int, int);
    extern int   __KAI_KMPC_CONVENTION  omp_target_associate_ptr(const void *, const void *, size_t, size_t, int);
    extern int   __KAI_KMPC_CONVENTION  omp_target_disassociate_ptr(const void *, int);
    /* Extension */
    extern void* __KAI_KMPC_CONVENTION  omp_target_alloc_host(size_t, int);
    extern void* __KAI_KMPC_CONVENTION  omp_target_alloc_shared(size_t, int);
    extern void* __KAI_KMPC_CONVENTION  omp_target_alloc_device(size_t, int);
    extern void* __KAI_KMPC_CONVENTION  omp_target_get_context(int);

    /* OpenMP 5.0 */
    extern int   __KAI_KMPC_CONVENTION  omp_get_device_num (void);
    typedef void * omp_depend_t;

    /* OpenMP 5.1 interop */
    typedef intptr_t omp_intptr_t;

    extern void __KAI_KMPC_CONVENTION ompx_dump_mapping_tables(void);

    /* 0..omp_get_num_interop_properties()-1 are reserved for implementation-defined properties */
    typedef enum omp_interop_property {
        omp_ipr_fr_id = -1,
        omp_ipr_fr_name = -2,
        omp_ipr_vendor = -3,
        omp_ipr_vendor_name = -4,
        omp_ipr_device_num = -5,
        omp_ipr_platform = -6,
        omp_ipr_device = -7,
        omp_ipr_device_context = -8,
        omp_ipr_targetsync = -9,
        omp_ipr_first = -9,
        omp_ipr_device_num_eus = 0,
        omp_ipr_device_num_threads_per_eu = 1,
        omp_ipr_device_eu_simd_width = 2,
        omp_ipr_device_num_eus_per_subslice = 3,
        omp_ipr_device_num_subslices_per_slice = 4,
        omp_ipr_device_num_slices = 5,
        omp_ipr_device_local_mem_size = 6,
        omp_ipr_device_global_mem_size = 7,
        omp_ipr_device_global_mem_cache_size = 8,
        omp_ipr_device_max_clock_frequency = 9,
        omp_ipr_is_imm_cmd_list = 10,
        omp_ipr_is_inorder = 11,
        omp_ipr_device_global_free_mem_size = 12
    } omp_interop_property_t;

    #define omp_interop_none 0

    typedef enum omp_interop_rc {
        omp_irc_no_value = 1,
        omp_irc_success = 0,
        omp_irc_empty = -1,
        omp_irc_out_of_range = -2,
        omp_irc_type_int = -3,
        omp_irc_type_ptr = -4,
        omp_irc_type_str = -5,
        omp_irc_other = -6
    } omp_interop_rc_t;

    typedef enum omp_interop_fr {
        omp_ifr_cuda = 1,
        omp_ifr_cuda_driver = 2,
        omp_ifr_opencl = 3,
        omp_ifr_sycl = 4,
        omp_ifr_hip = 5,
        omp_ifr_level_zero = 6,
        omp_ifr_unified_runtime = 7,
        omp_ifr_last = 8
    } omp_interop_fr_t;

    typedef void * omp_interop_t;

    /*!
     * The `omp_get_num_interop_properties` routine retrieves the number of implementation-defined properties available for an `omp_interop_t` object.
     */
    extern int          __KAI_KMPC_CONVENTION  omp_get_num_interop_properties(const omp_interop_t);
    /*!
     * The `omp_get_interop_int` routine retrieves an integer property from an `omp_interop_t` object.
     */
    extern omp_intptr_t __KAI_KMPC_CONVENTION  omp_get_interop_int(const omp_interop_t, omp_interop_property_t, int *);
    /*!
     * The `omp_get_interop_ptr` routine retrieves a pointer property from an `omp_interop_t` object.
     */
    extern void *       __KAI_KMPC_CONVENTION  omp_get_interop_ptr(const omp_interop_t, omp_interop_property_t, int *);
    /*!
     * The `omp_get_interop_str` routine retrieves a string property from an `omp_interop_t` object.
     */
    extern const char * __KAI_KMPC_CONVENTION  omp_get_interop_str(const omp_interop_t, omp_interop_property_t, int *);
    /*!
     * The `omp_get_interop_name` routine retrieves a property name from an `omp_interop_t` object.
     */
    extern const char * __KAI_KMPC_CONVENTION  omp_get_interop_name(const omp_interop_t, omp_interop_property_t);
    /*!
     * The `omp_get_interop_type_desc` routine retrieves a description of the type of a property associated with an `omp_interop_t` object.
     */
    extern const char * __KAI_KMPC_CONVENTION  omp_get_interop_type_desc(const omp_interop_t, omp_interop_property_t);
    /*!
     * The `omp_get_interop_rc_desc` routine retrieves a description of the return code associated with an `omp_interop_t` object.
     */
    extern const char * __KAI_KMPC_CONVENTION  omp_get_interop_rc_desc(const omp_interop_t, omp_interop_rc_t);

    /* OpenMP 5.1 device memory routines */
    extern int    __KAI_KMPC_CONVENTION  omp_target_memcpy_async(void *, const void *, size_t, size_t, size_t, int,
                                             int, int, omp_depend_t *);
    extern int    __KAI_KMPC_CONVENTION  omp_target_memcpy_rect_async(void *, const void *, size_t, int, const size_t *,
                                             const size_t *, const size_t *, const size_t *, const size_t *, int, int,
                                             int, omp_depend_t *);
    extern void * __KAI_KMPC_CONVENTION  omp_get_mapped_ptr(const void *, int);
    extern int    __KAI_KMPC_CONVENTION  omp_target_is_accessible(const void *, size_t, int);

    /* OpenMP 6.0 device memory routines */
    extern void * __KAI_KMPC_CONVENTION  omp_target_memset(void *, int, size_t, int);
    extern void * __KAI_KMPC_CONVENTION  omp_target_memset_async(void *, int, size_t, int, int, omp_depend_t *);

    /* Device enumerators */
    #define omp_initial_device ((int)-1)
    #define omp_invalid_device ((int)-10)

    /* kmp API functions */
    extern int    __KAI_KMPC_CONVENTION  kmp_get_stacksize          (void);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_stacksize          (int);
    extern size_t __KAI_KMPC_CONVENTION  kmp_get_stacksize_s        (void);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_stacksize_s        (size_t);
    extern int    __KAI_KMPC_CONVENTION  kmp_get_blocktime          (void);
    extern int    __KAI_KMPC_CONVENTION  kmp_get_library            (void);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_blocktime          (int);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_library            (int);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_library_serial     (void);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_library_turnaround (void);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_library_throughput (void);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_defaults           (char const *);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_disp_num_buffers   (int);

    /* Intel affinity API */
    typedef void * kmp_affinity_mask_t;

    extern int    __KAI_KMPC_CONVENTION  kmp_set_affinity             (kmp_affinity_mask_t *);
    extern int    __KAI_KMPC_CONVENTION  kmp_get_affinity             (kmp_affinity_mask_t *);
    extern int    __KAI_KMPC_CONVENTION  kmp_get_affinity_max_proc    (void);
    extern void   __KAI_KMPC_CONVENTION  kmp_create_affinity_mask     (kmp_affinity_mask_t *);
    extern void   __KAI_KMPC_CONVENTION  kmp_destroy_affinity_mask    (kmp_affinity_mask_t *);
    extern int    __KAI_KMPC_CONVENTION  kmp_set_affinity_mask_proc   (int, kmp_affinity_mask_t *);
    extern int    __KAI_KMPC_CONVENTION  kmp_unset_affinity_mask_proc (int, kmp_affinity_mask_t *);
    extern int    __KAI_KMPC_CONVENTION  kmp_get_affinity_mask_proc   (int, kmp_affinity_mask_t *);

    /* OpenMP 4.0 affinity API */
    typedef enum omp_proc_bind_t {
        omp_proc_bind_false = 0,
        omp_proc_bind_true = 1,
        omp_proc_bind_master = 2,
        omp_proc_bind_close = 3,
        omp_proc_bind_spread = 4
    } omp_proc_bind_t;

    extern omp_proc_bind_t __KAI_KMPC_CONVENTION omp_get_proc_bind (void);

    /* OpenMP 4.5 affinity API */
    extern int  __KAI_KMPC_CONVENTION omp_get_num_places (void);
    extern int  __KAI_KMPC_CONVENTION omp_get_place_num_procs (int);
    extern void __KAI_KMPC_CONVENTION omp_get_place_proc_ids (int, int *);
    extern int  __KAI_KMPC_CONVENTION omp_get_place_num (void);
    extern int  __KAI_KMPC_CONVENTION omp_get_partition_num_places (void);
    extern void __KAI_KMPC_CONVENTION omp_get_partition_place_nums (int *);

    extern void * __KAI_KMPC_CONVENTION  kmp_malloc  (size_t);
    extern void * __KAI_KMPC_CONVENTION  kmp_aligned_malloc  (size_t, size_t);
    extern void * __KAI_KMPC_CONVENTION  kmp_calloc  (size_t, size_t);
    extern void * __KAI_KMPC_CONVENTION  kmp_realloc (void *, size_t);
    extern void   __KAI_KMPC_CONVENTION  kmp_free    (void *);

    extern void   __KAI_KMPC_CONVENTION  kmp_set_warnings_on(void);
    extern void   __KAI_KMPC_CONVENTION  kmp_set_warnings_off(void);

    /* OpenMP 5.0 Tool Control */
    typedef enum omp_control_tool_result_t {
        omp_control_tool_notool = -2,
        omp_control_tool_nocallback = -1,
        omp_control_tool_success = 0,
        omp_control_tool_ignored = 1
    } omp_control_tool_result_t;

    typedef enum omp_control_tool_t {
        omp_control_tool_start = 1,
        omp_control_tool_pause = 2,
        omp_control_tool_flush = 3,
        omp_control_tool_end = 4
    } omp_control_tool_t;

    extern int __KAI_KMPC_CONVENTION omp_control_tool(int, int, void*);

    /* OpenMP 5.0 Memory Management */
    typedef uintptr_t omp_uintptr_t;

    typedef enum {
        omp_atk_sync_hint = 1,
        omp_atk_alignment = 2,
        omp_atk_access = 3,
        omp_atk_pool_size = 4,
        omp_atk_fallback = 5,
        omp_atk_fb_data = 6,
        omp_atk_pinned = 7,
        omp_atk_partition = 8,
        omp_atk_pin_device = 9,
        omp_atk_preferred_device = 10,
        omp_atk_device_access = 11,
        omp_atk_target_access = 12,
        omp_atk_atomic_scope = 13,
        omp_atk_part_size = 14
    } omp_alloctrait_key_t;

    typedef enum {
        omp_atv_false = 0,
        omp_atv_true = 1,
        omp_atv_contended = 3,
        omp_atv_uncontended = 4,
        omp_atv_serialized = 5,
        omp_atv_sequential = omp_atv_serialized, // (deprecated)
        omp_atv_private = 6,
        omp_atv_device = 7,
        omp_atv_thread = 8,
        omp_atv_pteam = 9,
        omp_atv_cgroup = 10,
        omp_atv_default_mem_fb = 11,
        omp_atv_null_fb = 12,
        omp_atv_abort_fb = 13,
        omp_atv_allocator_fb = 14,
        omp_atv_environment = 15,
        omp_atv_nearest = 16,
        omp_atv_blocked = 17,
        omp_atv_interleaved = 18,
        omp_atv_all = 19,
        omp_atv_single = 20,
        omp_atv_multiple = 21,
        omp_atv_memspace = 22
    } omp_alloctrait_value_t;
    #define omp_atv_default ((omp_uintptr_t)-1)

    typedef struct {
        omp_alloctrait_key_t key;
        omp_uintptr_t value;
    } omp_alloctrait_t;

#   if defined(_WIN32)
    // On Windows cl and icl do not support 64-bit enum, let's use integer then.
    typedef omp_uintptr_t omp_allocator_handle_t;
    extern __KMP_IMP omp_allocator_handle_t const omp_null_allocator;
    extern __KMP_IMP omp_allocator_handle_t const omp_default_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_large_cap_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_const_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_high_bw_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_low_lat_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_cgroup_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_pteam_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_thread_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_target_host_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_target_shared_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const omp_target_device_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const llvm_omp_target_host_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const llvm_omp_target_shared_mem_alloc;
    extern __KMP_IMP omp_allocator_handle_t const llvm_omp_target_device_mem_alloc;

    typedef omp_uintptr_t omp_memspace_handle_t;
    extern __KMP_IMP omp_memspace_handle_t const omp_null_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const omp_default_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const omp_large_cap_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const omp_const_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const omp_high_bw_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const omp_low_lat_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const omp_target_host_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const omp_target_shared_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const omp_target_device_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const llvm_omp_target_host_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const llvm_omp_target_shared_mem_space;
    extern __KMP_IMP omp_memspace_handle_t const llvm_omp_target_device_mem_space;
#   else
#       if __cplusplus >= 201103
    typedef enum omp_allocator_handle_t : omp_uintptr_t
#       else
    typedef enum omp_allocator_handle_t
#       endif
    {
      omp_null_allocator = 0,
      omp_default_mem_alloc = 1,
      omp_large_cap_mem_alloc = 2,
      omp_const_mem_alloc = 3,
      omp_high_bw_mem_alloc = 4,
      omp_low_lat_mem_alloc = 5,
      omp_cgroup_mem_alloc = 6,
      omp_pteam_mem_alloc = 7,
      omp_thread_mem_alloc = 8,
      llvm_omp_target_host_mem_alloc = 100,
      llvm_omp_target_shared_mem_alloc = 101,
      llvm_omp_target_device_mem_alloc = 102,
      omp_target_host_mem_alloc = llvm_omp_target_host_mem_alloc,
      omp_target_shared_mem_alloc = llvm_omp_target_shared_mem_alloc,
      omp_target_device_mem_alloc = llvm_omp_target_device_mem_alloc,
      KMP_ALLOCATOR_MAX_HANDLE = UINTPTR_MAX
    } omp_allocator_handle_t;
#       if __cplusplus >= 201103
    typedef enum omp_memspace_handle_t : omp_uintptr_t
#       else
    typedef enum omp_memspace_handle_t
#       endif
    {
      omp_null_mem_space = 0,
      omp_default_mem_space = 99,
      omp_large_cap_mem_space = 1,
      omp_const_mem_space = 2,
      omp_high_bw_mem_space = 3,
      omp_low_lat_mem_space = 4,
      llvm_omp_target_host_mem_space = 100,
      llvm_omp_target_shared_mem_space = 101,
      llvm_omp_target_device_mem_space = 102,
      omp_target_host_mem_space = llvm_omp_target_host_mem_space,
      omp_target_shared_mem_space = llvm_omp_target_shared_mem_space,
      omp_target_device_mem_space = llvm_omp_target_device_mem_space,
      KMP_MEMSPACE_MAX_HANDLE = UINTPTR_MAX
    } omp_memspace_handle_t;
#   endif
    extern omp_allocator_handle_t __KAI_KMPC_CONVENTION omp_init_allocator(omp_memspace_handle_t m,
                                                       int ntraits, omp_alloctrait_t traits[]);
    extern void __KAI_KMPC_CONVENTION omp_destroy_allocator(omp_allocator_handle_t allocator);

    extern void __KAI_KMPC_CONVENTION omp_set_default_allocator(omp_allocator_handle_t a);
    extern omp_allocator_handle_t __KAI_KMPC_CONVENTION omp_get_default_allocator(void);
#ifdef __cplusplus
    extern void *__KAI_KMPC_CONVENTION omp_alloc(size_t size, omp_allocator_handle_t a = omp_null_allocator);
    extern void *__KAI_KMPC_CONVENTION omp_aligned_alloc(size_t align, size_t size,
                                                         omp_allocator_handle_t a = omp_null_allocator);
    extern void *__KAI_KMPC_CONVENTION omp_calloc(size_t nmemb, size_t size,
                                                  omp_allocator_handle_t a = omp_null_allocator);
    extern void *__KAI_KMPC_CONVENTION omp_aligned_calloc(size_t align, size_t nmemb, size_t size,
                                                          omp_allocator_handle_t a = omp_null_allocator);
    extern void *__KAI_KMPC_CONVENTION omp_realloc(void *ptr, size_t size,
                                                   omp_allocator_handle_t allocator = omp_null_allocator,
                                                   omp_allocator_handle_t free_allocator = omp_null_allocator);
    extern void __KAI_KMPC_CONVENTION omp_free(void * ptr, omp_allocator_handle_t a = omp_null_allocator);
#else
    extern void *__KAI_KMPC_CONVENTION omp_alloc(size_t size, omp_allocator_handle_t a);
    extern void *__KAI_KMPC_CONVENTION omp_aligned_alloc(size_t align, size_t size,
                                                         omp_allocator_handle_t a);
    extern void *__KAI_KMPC_CONVENTION omp_calloc(size_t nmemb, size_t size, omp_allocator_handle_t a);
    extern void *__KAI_KMPC_CONVENTION omp_aligned_calloc(size_t align, size_t nmemb, size_t size,
                                                          omp_allocator_handle_t a);
    extern void *__KAI_KMPC_CONVENTION omp_realloc(void *ptr, size_t size, omp_allocator_handle_t allocator,
                                                   omp_allocator_handle_t free_allocator);
    extern void __KAI_KMPC_CONVENTION omp_free(void *ptr, omp_allocator_handle_t a);
#endif

    /* OpenMP TR11 routines to get memory spaces and allocators */
    extern omp_memspace_handle_t omp_get_devices_memspace(int ndevs, const int *devs, omp_memspace_handle_t memspace);
    extern omp_memspace_handle_t omp_get_device_memspace(int dev, omp_memspace_handle_t memspace);
    extern omp_memspace_handle_t omp_get_devices_and_host_memspace(int ndevs, const int *devs, omp_memspace_handle_t memspace);
    extern omp_memspace_handle_t omp_get_device_and_host_memspace(int dev, omp_memspace_handle_t memspace);
    extern omp_memspace_handle_t omp_get_devices_all_memspace(omp_memspace_handle_t memspace);
    extern omp_allocator_handle_t omp_get_devices_allocator(int ndevs, const int *devs, omp_memspace_handle_t memspace);
    extern omp_allocator_handle_t omp_get_device_allocator(int dev, omp_memspace_handle_t memspace);
    extern omp_allocator_handle_t omp_get_devices_and_host_allocator(int ndevs, const int *devs, omp_memspace_handle_t memspace);
    extern omp_allocator_handle_t omp_get_device_and_host_allocator(int dev, omp_memspace_handle_t memspace);
    extern omp_allocator_handle_t omp_get_devices_all_allocator(omp_memspace_handle_t memspace);
    extern int omp_get_memspace_num_resources(omp_memspace_handle_t memspace);
    extern omp_memspace_handle_t omp_get_submemspace(omp_memspace_handle_t memspace, int num_resources, int *resources);

    /* OpenMP 5.0 Affinity Format */
    extern void __KAI_KMPC_CONVENTION omp_set_affinity_format(char const *);
    extern size_t __KAI_KMPC_CONVENTION omp_get_affinity_format(char *, size_t);
    extern void __KAI_KMPC_CONVENTION omp_display_affinity(char const *);
    extern size_t __KAI_KMPC_CONVENTION omp_capture_affinity(char *, size_t, char const *);

    /* OpenMP 5.0 events */
#   if defined(_WIN32)
    // On Windows cl and icl do not support 64-bit enum, let's use integer then.
    typedef omp_uintptr_t omp_event_handle_t;
#   else
    typedef enum omp_event_handle_t { KMP_EVENT_MAX_HANDLE = UINTPTR_MAX } omp_event_handle_t;
#   endif
    extern void __KAI_KMPC_CONVENTION omp_fulfill_event ( omp_event_handle_t event );

    /* OpenMP 5.0 Pause Resources */
    typedef enum omp_pause_resource_t {
      omp_pause_resume = 0,
      omp_pause_soft = 1,
      omp_pause_hard = 2,
      omp_pause_stop_tool = 3
    } omp_pause_resource_t;
    extern int __KAI_KMPC_CONVENTION omp_pause_resource(omp_pause_resource_t, int);
    extern int __KAI_KMPC_CONVENTION omp_pause_resource_all(omp_pause_resource_t);

    extern int __KAI_KMPC_CONVENTION omp_get_supported_active_levels(void);

    /* OpenMP 5.1 */
    extern void __KAI_KMPC_CONVENTION omp_set_num_teams(int num_teams);
    extern int __KAI_KMPC_CONVENTION omp_get_max_teams(void);
    extern void __KAI_KMPC_CONVENTION omp_set_teams_thread_limit(int limit);
    extern int __KAI_KMPC_CONVENTION omp_get_teams_thread_limit(void);
    extern void __KAI_KMPC_CONVENTION omp_display_env(int verbose);

    /* Optimization for omp_is_initial_device() */
#   if defined(_OPENMP) && defined(__clang__) && __clang_major__ > 11 && !defined(__APPLE__)
    #pragma omp begin declare variant match(device={kind(host)})
    static inline int omp_is_initial_device(void) { return 1; }
    #pragma omp end declare variant
    #pragma omp begin declare variant match(device={kind(nohost)})
    static inline int omp_is_initial_device(void) { return 0; }
    #pragma omp end declare variant
#   endif

    /* OpenMP 5.2 */
    extern int __KAI_KMPC_CONVENTION omp_in_explicit_task(void);

    /* OpenMP 6.0 */
    extern int __KAI_KMPC_CONVENTION omp_is_free_agent(void);
    extern int __KAI_KMPC_CONVENTION omp_ancestor_is_free_agent(int);
    extern int __KAI_KMPC_CONVENTION omp_get_device_from_uid(const char *);
    extern const char * __KAI_KMPC_CONVENTION omp_get_uid_from_device(int);

    /* LLVM Extensions */
    extern void *llvm_omp_target_dynamic_shared_alloc(void);

    /* Extension for dynamic cgroup mem access on device */
    extern void *ompx_get_dyn_cgroup_mem(void);

    /* Extension for named barrier access in device code */
    extern void __KAI_KMPC_CONVENTION ompx_nbarrier_init(uint32_t nbarrier_count);
    extern void __KAI_KMPC_CONVENTION ompx_nbarrier_wait(uint32_t nbarrier_id);
    extern void __KAI_KMPC_CONVENTION ompx_nbarrier_signal(
        uint32_t nbarrier_id, uint32_t num_producers, uint32_t num_consumers, uint32_t op_type, uint32_t fence_type);

    /* Extension for target memory allocation */
    extern void* __KAI_KMPC_CONVENTION ompx_target_realloc(void *, size_t, int);
    extern void* __KAI_KMPC_CONVENTION ompx_target_realloc_device(void *, size_t, int);
    extern void* __KAI_KMPC_CONVENTION ompx_target_realloc_host(void *, size_t, int);
    extern void* __KAI_KMPC_CONVENTION ompx_target_realloc_shared(void *, size_t, int);
    extern void* __KAI_KMPC_CONVENTION ompx_target_aligned_alloc(size_t, size_t, int);
    extern void* __KAI_KMPC_CONVENTION ompx_target_aligned_alloc_device(size_t, size_t, int);
    extern void* __KAI_KMPC_CONVENTION ompx_target_aligned_alloc_host(size_t, size_t, int);
    extern void* __KAI_KMPC_CONVENTION ompx_target_aligned_alloc_shared(size_t, size_t, int);

    /* Extension for subdevice access */
    extern int __KAI_KMPC_CONVENTION ompx_get_num_subdevices(int device_num, int level);

    /* Extension for user-guided kernel batching */
    extern void __KAI_KMPC_CONVENTION ompx_kernel_batch_begin(int device_num, uint32_t max_kernels);
    extern void __KAI_KMPC_CONVENTION ompx_kernel_batch_end(int device_num);

    /* Extension for device information access */
    extern int __KAI_KMPC_CONVENTION ompx_get_device_info(
        int device_num, int info_id, size_t info_size, void *info_value, size_t *info_size_ret);
    enum kmp_device_info {
      ompx_devinfo_name = 0,
      ompx_devinfo_pci_id,
      ompx_devinfo_tile_id,
      ompx_devinfo_ccs_id,
      ompx_devinfo_num_eus,
      ompx_devinfo_num_threads_per_eu,
      ompx_devinfo_eu_simd_width,
      ompx_devinfo_num_eus_per_subslice,
      ompx_devinfo_num_subslices_per_slice,
      ompx_devinfo_num_slices,
      ompx_devinfo_local_mem_size,
      ompx_devinfo_global_mem_size,
      ompx_devinfo_global_mem_cache_size,
      ompx_devinfo_max_clock_frequency,
      ompx_devinfo_plugin_name,
      ompx_devinfo_max_mem_alloc_size
    };

    /* Extension for using GPU shared memory hint/prefetch */
    /* Note that these values are equivalent to L0 GPU RT definition except for ompx_mem_hint_none */
    enum kmp_mem_hint {
      ompx_mem_hint_read_mostly = 0,
      ompx_mem_hint_prefer_device = 2,
      ompx_mem_hint_non_atomic_mostly = 4,
      ompx_mem_hint_cached = 6,
      ompx_mem_hint_uncached = 7,
      ompx_mem_hint_prefer_host = 8,
      ompx_mem_hint_device_access_mostly = 9,
      ompx_mem_hint_host_access_mostly = 10,
      ompx_mem_hint_non_coherent = 11,
      ompx_mem_hint_none = UINT32_MAX
    };
    extern void * __KAI_KMPC_CONVENTION ompx_target_aligned_alloc_shared_with_hint(
        size_t align, size_t size, int access_hint, int device_num);
    extern int __KAI_KMPC_CONVENTION ompx_target_prefetch_shared_mem(
        size_t num_ptrs, void **ptrs, size_t *sizes, int device_num);

    /* Extension for registering host-allocated memory */
    extern int __KAI_KMPC_CONVENTION ompx_target_register_host_pointer(void *ptr, size_t size, int device_num);
    extern void __KAI_KMPC_CONVENTION ompx_target_unregister_host_pointer(void *ptr, int device_num);

    /* Extended API routine that returns device ID from a memory location */
    extern int __KAI_KMPC_CONVENTION ompx_get_device_from_ptr(const void *);

    /* Extended API routine that changes debug level of offload runtime */
    void __KAI_KMPC_CONVENTION ompx_set_libomptarget_debug(int);

#   undef __KAI_KMPC_CONVENTION
#   undef __KMP_IMP

    /* Warning:
       The following typedefs are not standard, deprecated and will be removed in a future release.
    */
    typedef int     omp_int_t;
    typedef double  omp_wtime_t;

#   ifdef __cplusplus
    }
#   endif

#endif /* __OMP_H */
