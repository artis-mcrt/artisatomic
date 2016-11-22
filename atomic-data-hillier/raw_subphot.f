!
! Subroutine to compute the photoionization cross-section for XzV. The
! returned cross-sections are in MB. They are for plotting purposes --
! not for use with CMFGEN.
!
!***************************
!***************************
! Currently implemented fits.
!
!     1 - Seaton formula fit [sigma_o,alpha,beta]
!     2 - Hydrogenic: Split l (z states, n > 11)
!     3 - Hydrogenic: Pure n level (all l, n >= 13)
!     4 - Used for CIV rates from Leobowitz (JQSRT 1972,12,299)
!     5 - Opacity project fits (from Peach, Sraph, and Seaton (1988)
!     6 - Hummer fits to the opacity cross-sections for HeI
!     7 - Modifed Seaton fit --- cross-section zero until offset edge.
!     8 - Modifed Hydrogenic split l: cross-section zero until offset edge.
!     9 - Verner ground state fits (multiple shells).
!
!******************************************************************************
!******************************************************************************
!
	SUBROUTINE RAW_SUBPHOT(PHOT,FREQ,CROSS_A,CROSS_TYPE,NCROSS,
	1              GS_EDGE,EXC_FREQ,ZION,AMASS,NCF)
	USE HYD_BF_PHOT_DATA
	IMPLICIT NONE
!
! Altered 17-Sep-2010 : Altered implementation of Verner ground-state fits
!
	INTEGER NCF
	INTEGER NCROSS
	REAL*8 FREQ(NCF)		!Cross-section
	REAL*8 PHOT(NCF)		!Cross-section
	REAL*8 GS_EDGE			!Energy for ionization to Ground State!
	REAL*8 EXC_FREQ
	REAL*8 AMASS
	REAL*8 ZION
!
	INTEGER CROSS_TYPE
	REAL*8 CROSS_A(10)
!
! External functions.
!
	INTEGER ERROR_LU
	REAL*8 VOIGT,HYDCROSSL,GBF
	EXTERNAL VOIGT,HYDCROSSL,GBF,ERROR_LU
!
! Common block with opacity/emissivity constants.
!
	REAL*8 CHIBF,CHIFF,HDKT,TWOHCSQ,OPLIN,EMLIN
	COMMON/CONSTANTS/ CHIBF,CHIFF,HDKT,TWOHCSQ
	COMMON/LINE/ OPLIN,EMLIN
!
! Local variables.
!
	INTEGER ML
	INTEGER I,J,K,l,N
	INTEGER LMIN
	INTEGER LST,LEND
!
	REAL*8, PARAMETER :: CONV_FAC=1.0D-08
	REAL*8, PARAMETER :: LG10_CONV_FAC=-8.0D0
!
	REAL*8 U			!Defined as FREQ/EDGE
	REAL*8 RU			!Defined as EDGE/FREQ
	REAL*8 NEF
!
	REAL*8 EDGE			!Ionization energy
	REAL*8 ALPHA_BF
	REAL*8 DELF
	REAL*8 T1,T2,T3
	REAL*8 DOP_NU
	REAL*8 X
	REAL*8 RJ
	REAL*8 SUM
	REAL*8 RYD_HZ
	REAL*8 EV_TO_HZ
!
	INTEGER, PARAMETER :: IZERO=0
	LOGICAL, PARAMETER :: L_TRUE=.TRUE.
	LOGICAL, PARAMETER :: L_FALSE=.FALSE.
!
	ALPHA_BF=2.815D-06*(ZION**4)
	EDGE = GS_EDGE + EXC_FREQ
	DELF=LOG(MAX(100.0D0,10.0D0*EDGE))/(NCF-1)
	PHOT(1:NCF)=0.0D0
!
	IF(CROSS_TYPE .EQ. 8)THEN
	   T1=EDGE+CROSS_A(4)
	ELSE IF(EDGE .LE. 0)THEN
	   T1=1.0D0
	ELSE
	   T1=EDGE
	END IF
	DO ML=1,NCF
	  FREQ(ML)=T1*EXP((ML-1)*DELF)
	END DO
	RYD_HZ=109737.31534D0*2.99792458D+10/(1.0D0+1.0D0/1840.0D0/AMASS)/1.0D+15
	IF(EDGE .GT. 0)THEN
	   NEF=ZION*SQRT(RYD_HZ/EDGE)
	ELSE
	   NEF=20.0D0
	END IF
!
! Seaton fit.
!
	IF(CROSS_TYPE .EQ. 1)THEN
	  IF(CROSS_A(1) .EQ. 0.0D0)THEN
	    PHOT(1:NCF)=0.0D0
	  ELSE
	    DO ML=1,NCF
	      RU=EDGE/FREQ(ML)
	      PHOT(ML)=CONV_FAC*
	1        CROSS_A(1)*( CROSS_A(2) + (1.0D0-CROSS_A(2))*RU )*( RU**CROSS_A(3) )
	    END DO
	  END IF
!
	ELSE IF(CROSS_TYPE .EQ. 2)THEN
!
! N, LST and LEND must be integer - all other values are double precision.
! If it wasn't for the loop over l, this could be done in the main loop.
!
! BF_CROSS contains the LOG10 hydrogenic cross-section.
!
	  LMIN=1
	  N=NINT( CROSS_A(LMIN) )
	  LST=NINT( CROSS_A(LMIN+1) )
	  LEND=NINT( CROSS_A(LMIN+2) )
!
	  DO ML=1,NCF
	    U=FREQ(ML)/EDGE
	    X=LOG10(U)
	    RJ=X/L_DEL_U
	    J=RJ
	    T1=RJ-J
	    J=J+1
	    SUM=0.0D0
	    DO L=LST,LEND
	      IF(J .LT. N_PER_L)THEN
	        J=J+BF_L_INDX(N,L)-1
	        T2=T1*BF_L_CROSS(J+1)+(1.0D0-T1)*BF_L_CROSS(J)
	      ELSE
	        J=BF_L_INDX(N,L)+N_PER_L-1
	        T2=(BF_L_CROSS(J)-BF_L_CROSS(J-1))*
	1                   (RJ-N_PER_L)+BF_L_CROSS(J)
	      END IF
	      SUM=SUM+(2*L+1)*(10.0D0**T2)
	      J=RJ+1 		!Restore as corrupted
	    END DO
	    SUM=SUM*( NEF/(N*ZION) )**2
	    PHOT(ML)=SUM/( (LEND-LST+1)*(LEND+LST+1) )
	  END DO
!
! Hydrogenic transitions.
	ELSE IF(CROSS_TYPE .EQ. 3)THEN
!
! HYD_N_DATA contains the Bound-free gaunt factor.
!
	  N=CROSS_A(2)
	  DO ML=1,NCF
	    U=FREQ(ML)/EDGE
	    X=LOG10(U)
	    RJ=X/N_DEL_U
	    J=RJ
	    T1=RJ-J
	    J=J+1
	    IF(J .LT. N_PER_N)THEN
	      J=J+BF_N_INDX(N)-1
	      T1=T1*BF_N_GAUNT(J+1)+(1.0D0-T1)*BF_N_GAUNT(J)
	    ELSE
!
! Power law extrapolation.
!
	      J=BF_N_INDX(N)+N_PER_N-1
	      T1=LOG10(BF_N_GAUNT(J-1)/BF_N_GAUNT(J))
1             T1=BF_N_GAUNT(J)*( 10.0D0**(T1*(N_PER_N-RJ)) )
	    END IF
!
! NB: ZION is already include in ALPHA_BF
!
	    PHOT(ML)=ALPHA_BF*T1*CROSS_A(1)/NEF/N/( (FREQ(ML)*NEF)**3 )
	  END DO
!
! Used for CIV recombination rates for s and p states.
!(ref -Leibowitz J.Q.S.R.T 1972,12,299)
!
	ELSE IF(CROSS_TYPE .EQ. 4)THEN
	  DO ML=1,NCF
	    RU=EDGE/FREQ(ML)
	    T1=CONV_FAC*(  CROSS_A(1)+RU*( CROSS_A(2) +
	1        RU*(CROSS_A(3) + RU*(CROSS_A(4)+
	1        RU*(CROSS_A(5)+RU*CROSS_A(6)))) )  )
	    PHOT(ML)=T1
	  END DO
!
	ELSE IF(CROSS_TYPE .EQ. 5)THEN
!
! These fits are fits to the opacity cross section. Data taken from opacity
! project - Peach, Saraph, and Seaton (1988, C J. Phys. B: 21, 3669-3683)
!
! If the cross-section lies outside the range given by the fits, we assume
! that it scales as nu^{-2}. The values in this region should be unimportant.
!
	  DO ML=1,NCF
	    U=FREQ(ML)/EDGE
	    X=MIN( U,CROSS_A(5) )
	    X=DLOG10(X)
	    T1=10**(  CROSS_A(1)+X*( CROSS_A(2) + X*(CROSS_A(3) +
	1               X*CROSS_A(4)) ) + LG10_CONV_FAC  )
	    IF(U .GT. CROSS_A(5))T1=T1*(CROSS_A(LMIN+5)/U)**2
	    PHOT(ML)=T1
	  END DO
!
	ELSE IF(CROSS_TYPE .EQ. 6)THEN
!
! This type is for the Hummer fits to the Opacity cross sections of HeI.
! See HEI_PHOT_OPAC.
!
! We issue the SPACING command to ensure that rounding error does not
! cause X to be < 0 at the bound-free edge.
!
! U+SPACING(U) is the smallest number different from U (and lager).
!
	  LMIN=1
	  DO ML=1,NCF
	    U=FREQ(ML)/EDGE
	    X=LOG10(U+3.0D0*SPACING(U))
	    IF(X .GE. 0)THEN
              IF(X .LT. CROSS_A(LMIN+4))THEN
	        T1=((CROSS_A(LMIN+3)*X+CROSS_A(LMIN+2))*X +
	1                 CROSS_A(LMIN+1))*X+CROSS_A(LMIN)
	      ELSE
                T1=CROSS_A(LMIN+5)+CROSS_A(LMIN+6)*X
	      END IF
	      PHOT(ML)=10.0D0**(T1+LG10_CONV_FAC)
	    END IF
	  END DO
!
! Modified seaton fit.
!
	ELSE IF(CROSS_TYPE .EQ. 7 .AND. CROSS_A(1) .NE. 0)THEN
	    LMIN=1
	    DO ML=1,NCF
	      RU=EDGE/(FREQ(ML)+CROSS_A(LMIN+3))
	      IF(RU .LE. 1.0D0)THEN
	        PHOT(ML)=CONV_FAC*CROSS_A(LMIN)*( CROSS_A(LMIN+1) +
	1               (1.0D0-CROSS_A(LMIN+1))*RU )*( RU**CROSS_A(LMIN+2) )
	      END IF
	    END DO
	ELSE IF(CROSS_TYPE .EQ. 8)THEN
!
! N, LST and LEND must be integer - all other values are double precision.
! If it wasn't for the loop over l, this could be done in the main loop.
!
! BF_CROSS contains the LOG10 hydrogenic cross-section.
!
	   LMIN=1
	   N=NINT( CROSS_A(LMIN) )
	   LST=NINT( CROSS_A(LMIN+1) )
	   LEND=NINT( CROSS_A(LMIN+2) )
	   DO ML=1,NCF
	     IF(FREQ(ML) .GE. EDGE+CROSS_A(LMIN+3))THEN
	       U=FREQ(ML)/(EDGE+CROSS_A(LMIN+3))
!
	       X=LOG10(U)
	       RJ=X/L_DEL_U
	       J=RJ
	       T1=RJ-J
	       J=J+1
	       SUM=0.0D0
	       DO L=LST,LEND
	         IF(J .LT. N_PER_L)THEN
	           J=J+BF_L_INDX(N,L)-1
	           T2=T1*BF_L_CROSS(J+1)+(1.0D0-T1)*BF_L_CROSS(J)
	         ELSE
	           J=BF_L_INDX(N,L)+N_PER_L-1
	           T2=(BF_L_CROSS(J)-BF_L_CROSS(J-1))*
	1                   (RJ-N_PER_L)+BF_L_CROSS(J)
	         END IF
	         SUM=SUM+(2*L+1)*(10.0D0**T2)
		 J=RJ+1 		!Restore as corrupted
	       END DO
	       SUM=SUM/ZION/ZION	!Ignore neff correction :( NEF(I,K)/(N*ZION) )**2
	       PHOT(ML)=SUM/( (LEND-LST+1)*(LEND+LST+1) )
	    END IF	!Above threshold
	  END DO
	ELSE IF(CROSS_TYPE .EQ. 9)THEN
!
	   EV_TO_HZ=0.241798840766D0
	   WRITE(6,*)NCROSS
	   WRITE(6,*)GS_EDGE,EV_TO_HZ*CROSS_A(4)
	   WRITE(6,*)CROSS_A(1:9)
	   K=1
	   DO I=1,NCROSS/8
	     IF(I .NE. 1)EDGE=EV_TO_HZ*CROSS_A(K+2)
	     DO ML=1,NCF
	      IF(FREQ(ML) .GE. EDGE)THEN
	        U=FREQ(ML)/CROSS_A(K+3)/EV_TO_HZ
	        T1=(U-1.0D0)**2 + CROSS_A(K+7)**2
	        T2=U**( 5.5D0+CROSS_A(K+1)-0.5D0*CROSS_A(K+6) )
	        T3=( 1.0D0+SQRT(U/CROSS_A(K+5)) )**CROSS_A(K+6)
	        PHOT(ML)=PHOT(ML)+1.0D-08*T1*CROSS_A(K+4)/T2/T3
	      END IF
	    END DO
	    K=K+8
	  END DO
	ELSE
!
	  WRITE(6,*)'Unrecognized photoionization cross-section type'
	  WRITE(6,*)'Cross-section type=',CROSS_TYPE
!
! More cross-section types can be added in here.
!
	END IF		!Type of cross-section
!
! Convert from CMFGEN units to MB.
!
	PHOT(1:NCF)=1.0D+08*PHOT(1:NCF)
!
	RETURN
	END
