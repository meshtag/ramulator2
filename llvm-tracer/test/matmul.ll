; ModuleID = 'test/matmul.c'
source_filename = "test/matmul.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

@A = common global [64 x [64 x float]] zeroinitializer, align 4
@B = common global [64 x [64 x float]] zeroinitializer, align 4
@C = common global [64 x [64 x float]] zeroinitializer, align 4

; Function Attrs: noinline nounwind ssp uwtable(sync)
define i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  store i32 0, ptr %2, align 4
  br label %7

7:                                                ; preds = %45, %0
  %8 = load i32, ptr %2, align 4
  %9 = icmp slt i32 %8, 64
  br i1 %9, label %10, label %48

10:                                               ; preds = %7
  store i32 0, ptr %3, align 4
  br label %11

11:                                               ; preds = %41, %10
  %12 = load i32, ptr %3, align 4
  %13 = icmp slt i32 %12, 64
  br i1 %13, label %14, label %44

14:                                               ; preds = %11
  %15 = load i32, ptr %2, align 4
  %16 = load i32, ptr %3, align 4
  %17 = add nsw i32 %15, %16
  %18 = sitofp i32 %17 to float
  %19 = load i32, ptr %2, align 4
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds [64 x [64 x float]], ptr @A, i64 0, i64 %20
  %22 = load i32, ptr %3, align 4
  %23 = sext i32 %22 to i64
  %24 = getelementptr inbounds [64 x float], ptr %21, i64 0, i64 %23
  store float %18, ptr %24, align 4
  %25 = load i32, ptr %2, align 4
  %26 = load i32, ptr %3, align 4
  %27 = sub nsw i32 %25, %26
  %28 = sitofp i32 %27 to float
  %29 = load i32, ptr %2, align 4
  %30 = sext i32 %29 to i64
  %31 = getelementptr inbounds [64 x [64 x float]], ptr @B, i64 0, i64 %30
  %32 = load i32, ptr %3, align 4
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds [64 x float], ptr %31, i64 0, i64 %33
  store float %28, ptr %34, align 4
  %35 = load i32, ptr %2, align 4
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds [64 x [64 x float]], ptr @C, i64 0, i64 %36
  %38 = load i32, ptr %3, align 4
  %39 = sext i32 %38 to i64
  %40 = getelementptr inbounds [64 x float], ptr %37, i64 0, i64 %39
  store float 0.000000e+00, ptr %40, align 4
  br label %41

41:                                               ; preds = %14
  %42 = load i32, ptr %3, align 4
  %43 = add nsw i32 %42, 1
  store i32 %43, ptr %3, align 4
  br label %11, !llvm.loop !6

44:                                               ; preds = %11
  br label %45

45:                                               ; preds = %44
  %46 = load i32, ptr %2, align 4
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr %2, align 4
  br label %7, !llvm.loop !8

48:                                               ; preds = %7
  store i32 0, ptr %4, align 4
  br label %49

49:                                               ; preds = %91, %48
  %50 = load i32, ptr %4, align 4
  %51 = icmp slt i32 %50, 64
  br i1 %51, label %52, label %94

52:                                               ; preds = %49
  store i32 0, ptr %5, align 4
  br label %53

53:                                               ; preds = %87, %52
  %54 = load i32, ptr %5, align 4
  %55 = icmp slt i32 %54, 64
  br i1 %55, label %56, label %90

56:                                               ; preds = %53
  store i32 0, ptr %6, align 4
  br label %57

57:                                               ; preds = %83, %56
  %58 = load i32, ptr %6, align 4
  %59 = icmp slt i32 %58, 64
  br i1 %59, label %60, label %86

60:                                               ; preds = %57
  %61 = load i32, ptr %4, align 4
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds [64 x [64 x float]], ptr @A, i64 0, i64 %62
  %64 = load i32, ptr %6, align 4
  %65 = sext i32 %64 to i64
  %66 = getelementptr inbounds [64 x float], ptr %63, i64 0, i64 %65
  %67 = load float, ptr %66, align 4
  %68 = load i32, ptr %6, align 4
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds [64 x [64 x float]], ptr @B, i64 0, i64 %69
  %71 = load i32, ptr %5, align 4
  %72 = sext i32 %71 to i64
  %73 = getelementptr inbounds [64 x float], ptr %70, i64 0, i64 %72
  %74 = load float, ptr %73, align 4
  %75 = load i32, ptr %4, align 4
  %76 = sext i32 %75 to i64
  %77 = getelementptr inbounds [64 x [64 x float]], ptr @C, i64 0, i64 %76
  %78 = load i32, ptr %5, align 4
  %79 = sext i32 %78 to i64
  %80 = getelementptr inbounds [64 x float], ptr %77, i64 0, i64 %79
  %81 = load float, ptr %80, align 4
  %82 = call float @llvm.fmuladd.f32(float %67, float %74, float %81)
  store float %82, ptr %80, align 4
  br label %83

83:                                               ; preds = %60
  %84 = load i32, ptr %6, align 4
  %85 = add nsw i32 %84, 1
  store i32 %85, ptr %6, align 4
  br label %57, !llvm.loop !9

86:                                               ; preds = %57
  br label %87

87:                                               ; preds = %86
  %88 = load i32, ptr %5, align 4
  %89 = add nsw i32 %88, 1
  store i32 %89, ptr %5, align 4
  br label %53, !llvm.loop !10

90:                                               ; preds = %53
  br label %91

91:                                               ; preds = %90
  %92 = load i32, ptr %4, align 4
  %93 = add nsw i32 %92, 1
  store i32 %93, ptr %4, align 4
  br label %49, !llvm.loop !11

94:                                               ; preds = %49
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #1

attributes #0 = { noinline nounwind ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+bti,+ccdp,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
