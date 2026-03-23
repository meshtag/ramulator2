; ModuleID = '/Users/meshtag/TritonPIM/third_party/ramulator2/llvm-tracer/test/axpy_pim.c'
source_filename = "/Users/meshtag/TritonPIM/third_party/ramulator2/llvm-tracer/test/axpy_pim.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

@.str = private unnamed_addr constant [14 x i8] c"pim_trace.txt\00", align 1
@__const.main.dims = private unnamed_addr constant [1 x i32] [i32 1024], align 4
@x = internal global [1024 x float] zeroinitializer, align 4
@y = internal global [1024 x float] zeroinitializer, align 4

; Function Attrs: noinline nounwind ssp uwtable(sync)
define i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca [1 x i32], align 4
  %3 = alloca i32, align 4
  %4 = alloca float, align 4
  %5 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  call void @pim_init(ptr noundef @.str)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %2, ptr align 4 @__const.main.dims, i64 4, i1 false)
  %6 = getelementptr inbounds [1 x i32], ptr %2, i64 0, i64 0
  %7 = call i32 @pim_register_tensor(ptr noundef @x, ptr noundef %6, i32 noundef 1, i32 noundef 4, i32 noundef 0)
  %8 = getelementptr inbounds [1 x i32], ptr %2, i64 0, i64 0
  %9 = call i32 @pim_register_tensor(ptr noundef @y, ptr noundef %8, i32 noundef 1, i32 noundef 4, i32 noundef 2)
  store i32 0, ptr %3, align 4
  br label %10

10:                                               ; preds = %28, %0
  %11 = load i32, ptr %3, align 4
  %12 = icmp slt i32 %11, 1024
  br i1 %12, label %13, label %31

13:                                               ; preds = %10
  %14 = load i32, ptr %3, align 4
  %15 = srem i32 %14, 97
  %16 = sitofp i32 %15 to float
  %17 = fmul float 0x3F847AE140000000, %16
  %18 = load i32, ptr %3, align 4
  %19 = sext i32 %18 to i64
  %20 = getelementptr inbounds [1024 x float], ptr @x, i64 0, i64 %19
  store float %17, ptr %20, align 4
  %21 = load i32, ptr %3, align 4
  %22 = srem i32 %21, 83
  %23 = sitofp i32 %22 to float
  %24 = fmul float 0x3F947AE140000000, %23
  %25 = load i32, ptr %3, align 4
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds [1024 x float], ptr @y, i64 0, i64 %26
  store float %24, ptr %27, align 4
  br label %28

28:                                               ; preds = %13
  %29 = load i32, ptr %3, align 4
  %30 = add nsw i32 %29, 1
  store i32 %30, ptr %3, align 4
  br label %10, !llvm.loop !6

31:                                               ; preds = %10
  store float 2.500000e+00, ptr %4, align 4
  call void @pim_set_phase(i32 noundef 1)
  store i32 0, ptr %5, align 4
  br label %32

32:                                               ; preds = %49, %31
  %33 = load i32, ptr %5, align 4
  %34 = icmp slt i32 %33, 1024
  br i1 %34, label %35, label %52

35:                                               ; preds = %32
  %36 = load float, ptr %4, align 4
  %37 = load i32, ptr %5, align 4
  %38 = sext i32 %37 to i64
  %39 = getelementptr inbounds [1024 x float], ptr @x, i64 0, i64 %38
  %40 = load float, ptr %39, align 4
  %41 = load i32, ptr %5, align 4
  %42 = sext i32 %41 to i64
  %43 = getelementptr inbounds [1024 x float], ptr @y, i64 0, i64 %42
  %44 = load float, ptr %43, align 4
  %45 = call float @llvm.fmuladd.f32(float %36, float %40, float %44)
  %46 = load i32, ptr %5, align 4
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds [1024 x float], ptr @y, i64 0, i64 %47
  store float %45, ptr %48, align 4
  br label %49

49:                                               ; preds = %35
  %50 = load i32, ptr %5, align 4
  %51 = add nsw i32 %50, 1
  store i32 %51, ptr %5, align 4
  br label %32, !llvm.loop !8

52:                                               ; preds = %32
  call void @pim_set_phase(i32 noundef 0)
  call void @pim_finalize()
  ret i32 0
}

declare void @pim_init(ptr noundef) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #2

declare i32 @pim_register_tensor(ptr noundef, ptr noundef, i32 noundef, i32 noundef, i32 noundef) #1

declare void @pim_set_phase(i32 noundef) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #3

declare void @pim_finalize() #1

attributes #0 = { noinline nounwind ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Apple clang version 17.0.0 (clang-1700.4.4.1)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
