; ModuleID = 'test/matmul.ll'
source_filename = "test/matmul.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

@A = common local_unnamed_addr global [64 x [64 x float]] zeroinitializer, align 4
@B = common local_unnamed_addr global [64 x [64 x float]] zeroinitializer, align 4
@C = common local_unnamed_addr global [64 x [64 x float]] zeroinitializer, align 4

; Function Attrs: nofree noinline nosync nounwind ssp memory(readwrite, argmem: none, inaccessiblemem: none) uwtable(sync)
define i32 @main() local_unnamed_addr #0 {
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(16384) @C, i8 0, i64 16384, i1 false)
  br label %vector.ph

vector.ph:                                        ; preds = %middle.block, %0
  %indvar = phi i64 [ 0, %0 ], [ %indvar.next, %middle.block ]
  %broadcast.splatinsert = insertelement <4 x i64> poison, i64 %indvar, i64 0
  %broadcast.splat = shufflevector <4 x i64> %broadcast.splatinsert, <4 x i64> poison, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.ind = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph ], [ %vec.ind.next, %vector.body ]
  %1 = add nuw nsw <4 x i64> %vec.ind, %broadcast.splat
  %2 = trunc <4 x i64> %1 to <4 x i32>
  %3 = sitofp <4 x i32> %2 to <4 x float>
  %4 = getelementptr inbounds [64 x [64 x float]], ptr @A, i64 0, i64 %indvar, i64 %index
  store <4 x float> %3, ptr %4, align 4
  %5 = sub nsw <4 x i64> %broadcast.splat, %vec.ind
  %6 = trunc <4 x i64> %5 to <4 x i32>
  %7 = sitofp <4 x i32> %6 to <4 x float>
  %8 = getelementptr inbounds [64 x [64 x float]], ptr @B, i64 0, i64 %indvar, i64 %index
  store <4 x float> %7, ptr %8, align 4
  %index.next = add nuw i64 %index, 4
  %vec.ind.next = add <4 x i64> %vec.ind, <i64 4, i64 4, i64 4, i64 4>
  %9 = icmp eq i64 %index.next, 64
  br i1 %9, label %middle.block, label %vector.body, !llvm.loop !6

middle.block:                                     ; preds = %vector.body
  %indvar.next = add nuw nsw i64 %indvar, 1
  %exitcond40.not = icmp eq i64 %indvar.next, 64
  br i1 %exitcond40.not, label %.preheader30, label %vector.ph, !llvm.loop !10

.preheader30:                                     ; preds = %middle.block, %19
  %indvars.iv49 = phi i64 [ %indvars.iv.next50, %19 ], [ 0, %middle.block ]
  br label %.preheader

.preheader:                                       ; preds = %.preheader30, %18
  %indvars.iv45 = phi i64 [ 0, %.preheader30 ], [ %indvars.iv.next46, %18 ]
  %10 = getelementptr inbounds [64 x [64 x float]], ptr @C, i64 0, i64 %indvars.iv49, i64 %indvars.iv45
  %.promoted = load float, ptr %10, align 4
  br label %11

11:                                               ; preds = %.preheader, %11
  %indvars.iv41 = phi i64 [ 0, %.preheader ], [ %indvars.iv.next42, %11 ]
  %12 = phi float [ %.promoted, %.preheader ], [ %17, %11 ]
  %13 = getelementptr inbounds [64 x [64 x float]], ptr @A, i64 0, i64 %indvars.iv49, i64 %indvars.iv41
  %14 = load float, ptr %13, align 4
  %15 = getelementptr inbounds [64 x [64 x float]], ptr @B, i64 0, i64 %indvars.iv41, i64 %indvars.iv45
  %16 = load float, ptr %15, align 4
  %17 = tail call float @llvm.fmuladd.f32(float %14, float %16, float %12)
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %exitcond44.not = icmp eq i64 %indvars.iv.next42, 64
  br i1 %exitcond44.not, label %18, label %11, !llvm.loop !11

18:                                               ; preds = %11
  store float %17, ptr %10, align 4
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond48.not = icmp eq i64 %indvars.iv.next46, 64
  br i1 %exitcond48.not, label %19, label %.preheader, !llvm.loop !12

19:                                               ; preds = %18
  %indvars.iv.next50 = add nuw nsw i64 %indvars.iv49, 1
  %exitcond52.not = icmp eq i64 %indvars.iv.next50, 64
  br i1 %exitcond52.not, label %20, label %.preheader30, !llvm.loop !13

20:                                               ; preds = %19
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #2

attributes #0 = { nofree noinline nosync nounwind ssp memory(readwrite, argmem: none, inaccessiblemem: none) uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: write) }

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
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
