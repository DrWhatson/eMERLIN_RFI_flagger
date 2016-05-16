subroutine weighted(arr, wei, na, nb, ker, ma, mb, smo)
  implicit none

  integer(4) :: na, nb, ma, mb
  real(8) :: arr(0:na,0:nb), wei(0:na,0:nb)
  real(8) :: ker(0:ma,0:mb)
  real(8), intent(out),dimension(0:na,0:nb) :: smo

  real(8) :: aw(0:na,0:nb)
  real(8) :: saw,sw
  integer(4) :: i, j, ilo, ihi, jlo, jhi, nx, ny
  integer(4) :: kxlo, kxhi, kylo, kyhi

!f2py integer(4), intent(in) na, nb, ma, mb
!f2py real(8), intent(in), dimension(0:na) :: arr
!f2py real(8), intent(in), dimension(0:na) :: wei
!f2py real(8), dimension(0:na,0:nb)  :: weighted

aw = arr*wei

nx = ma/2
ny = mb/2

do i=0,na
   do j=0,nb
      
      ilo = i-nx
      kxlo = 0
      if (ilo.le.0) then
         kxlo = -ilo
         ilo = 0
      endif
         
      ihi = i+nx
      kxhi = ma
      if (ihi.gt.na) then
         kxhi = ma -ihi + na
         ihi = na
      endif

      jlo = j-ny
      kylo = 0
      if (jlo.le.0) then
         kylo = -jlo
         jlo = 0
      endif

      jhi = j+ny
      kyhi = mb
      if (jhi.gt.nb) then
         kyhi = mb -jhi + nb
         jhi = nb
      endif
      
      saw = SUM(aw(ilo:ihi,jlo:jhi)*ker(kxlo:kxhi,kylo:kyhi))
      sw  = SUM(wei(ilo:ihi,jlo:jhi)*ker(kxlo:kxhi,kylo:kyhi))
      
      smo(i,j) = saw/sw

   enddo
enddo

end subroutine weighted
      

