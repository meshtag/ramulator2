; ModuleID = '/Users/meshtag/LLVMRamulator/ramulator2/llvm-tracer/.test_work/axpy_pim/axpy_pim.ll'
source_filename = "/Users/meshtag/LLVMRamulator/ramulator2/llvm-tracer/test/axpy_pim.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

@.str = private unnamed_addr constant [14 x i8] c"pim_trace.txt\00", align 1
@x = internal global [1024 x float] zeroinitializer, align 4
@y = internal global [1024 x float] zeroinitializer, align 4

; Function Attrs: noinline nounwind ssp uwtable(sync)
define i32 @main() local_unnamed_addr #0 {
vector.ph:
  %0 = alloca [1 x i32], align 4
  tail call void @pim_init(ptr noundef nonnull @.str) #3
  store i32 1024, ptr %0, align 4
  %1 = call i32 @pim_register_tensor(ptr noundef nonnull @x, ptr noundef nonnull %0, i32 noundef 1, i32 noundef 4, i32 noundef 0) #3
  %2 = call i32 @pim_register_tensor(ptr noundef nonnull @y, ptr noundef nonnull %0, i32 noundef 1, i32 noundef 4, i32 noundef 2) #3
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind = phi <4 x i16> [ <i16 0, i16 1, i16 2, i16 3>, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %3 = urem <4 x i16> %vec.ind, <i16 97, i16 97, i16 97, i16 97>
  %4 = uitofp <4 x i16> %3 to <4 x float>
  %5 = fmul <4 x float> %4, <float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000, float 0x3F847AE140000000>
  %6 = getelementptr inbounds [1024 x float], ptr @x, i64 0, i64 %index
  store <4 x float> %5, ptr %6, align 4
  %7 = urem <4 x i16> %vec.ind, <i16 83, i16 83, i16 83, i16 83>
  %8 = uitofp <4 x i16> %7 to <4 x float>
  %9 = fmul <4 x float> %8, <float 0x3F947AE140000000, float 0x3F947AE140000000, float 0x3F947AE140000000, float 0x3F947AE140000000>
  %10 = getelementptr inbounds [1024 x float], ptr @y, i64 0, i64 %index
  store <4 x float> %9, ptr %10, align 4
  %index.next = add nuw i64 %index, 4
  %vec.ind.next = add <4 x i16> %vec.ind, <i16 4, i16 4, i16 4, i16 4>
  %11 = icmp eq i64 %index.next, 1024
  br i1 %11, label %middle.block, label %vector.body, !llvm.loop !6

middle.block:                                     ; preds = %vector.body
  call void @pim_set_phase(i32 noundef 1) #3
  br label %vector.body25

vector.body25:                                    ; preds = %vector.body25, %middle.block
  %index26 = phi i64 [ 0, %middle.block ], [ %index.next34, %vector.body25 ]
  %12 = getelementptr inbounds [1024 x float], ptr @x, i64 0, i64 %index26
  %wide.load = load <4 x float>, ptr %12, align 4
  %13 = getelementptr inbounds float, ptr %12, i64 4
  %wide.load27 = load <4 x float>, ptr %13, align 4
  %14 = getelementptr inbounds float, ptr %12, i64 8
  %wide.load28 = load <4 x float>, ptr %14, align 4
  %15 = getelementptr inbounds float, ptr %12, i64 12
  %wide.load29 = load <4 x float>, ptr %15, align 4
  %16 = getelementptr inbounds [1024 x float], ptr @y, i64 0, i64 %index26
  %wide.load30 = load <4 x float>, ptr %16, align 4
  %17 = getelementptr inbounds float, ptr %16, i64 4
  %wide.load31 = load <4 x float>, ptr %17, align 4
  %18 = getelementptr inbounds float, ptr %16, i64 8
  %wide.load32 = load <4 x float>, ptr %18, align 4
  %19 = getelementptr inbounds float, ptr %16, i64 12
  %wide.load33 = load <4 x float>, ptr %19, align 4
  %20 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %wide.load, <4 x float> <float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00>, <4 x float> %wide.load30)
  %21 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %wide.load27, <4 x float> <float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00>, <4 x float> %wide.load31)
  %22 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %wide.load28, <4 x float> <float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00>, <4 x float> %wide.load32)
  %23 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %wide.load29, <4 x float> <float 2.500000e+00, float 2.500000e+00, float 2.500000e+00, float 2.500000e+00>, <4 x float> %wide.load33)
  store <4 x float> %20, ptr %16, align 4
  store <4 x float> %21, ptr %17, align 4
  store <4 x float> %22, ptr %18, align 4
  store <4 x float> %23, ptr %19, align 4
  %index.next34 = add nuw i64 %index26, 16
  %24 = icmp eq i64 %index.next34, 1024
  br i1 %24, label %middle.block21, label %vector.body25, !llvm.loop !10

middle.block21:                                   ; preds = %vector.body25
  call void @pim_set_phase(i32 noundef 0) #3
  call void @pim_finalize() #3
  ret i32 0
}

declare void @pim_init(ptr noundef) local_unnamed_addr #1

declare i32 @pim_register_tensor(ptr noundef, ptr noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #1

declare void @pim_set_phase(i32 noundef) local_unnamed_addr #1

declare void @pim_finalize() local_unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #2

attributes #0 = { noinline nounwind ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Apple clang version 17.0.0 (clang-1700.4.4.1)"}
!6 = distinct !{!6, !7, !8, !9}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!"llvm.loop.isvectorized", i32 1}
!9 = !{!"llvm.loop.unroll.runtime.disable"}
!10 = distinct !{!10, !7, !8, !9}
