! filepath: c:\Users\galac\Desktop\AMON\File_Read.f90
module my_module
    implicit none
    private
    public :: read_CM, read_mu

contains
    subroutine read_CM(filename, CM_x, CM_y, CM_z)
        character(len=*), intent(in) :: filename
        real, allocatable, intent(out) :: CM_x(:), CM_y(:), CM_z(:)
        integer :: io_status, n_lines, i
        
        ! First count the lines in file
        n_lines = 0
        open(unit=20, file=filename, status='old', action='read', iostat=io_status)
        if (io_status /= 0) then
            print *, "Error opening file: ", filename
            return
        end if
        
        do
            read(20, *, iostat=io_status)
            if (io_status /= 0) exit
            n_lines = n_lines + 1
        end do
        
        ! Allocate arrays and read data
        allocate(CM_x(n_lines), CM_y(n_lines), CM_z(n_lines))
        rewind(20)
        
        do i = 1, n_lines
            read(20, *) CM_x(i), CM_y(i), CM_z(i)
        end do
        
        close(20)
        
    end subroutine read_CM


    subroutine read_mu(filename, mu)
        character(len=*), intent(in) :: filename
        real, allocatable, intent(out) :: mu(:)
        integer :: io_status, n_lines, i
        
        ! First count the lines in file
        n_lines = 0
        open(unit=10, file=filename, status='old', action='read', iostat=io_status)
        if (io_status /= 0) then
            print *, "Error opening file: ", filename
            return
        end if
        
        do
            read(10, *, iostat=io_status)
            if (io_status /= 0) exit
            n_lines = n_lines + 1
        end do
        
        ! Allocate arrays and read data
        allocate(mu(n_lines))
        rewind(10)
        
        do i = 1, n_lines
            read(10, *) mu(i)
        end do
        
        close(10)
        


    end subroutine read_mu


end module my_module

program main
    use my_module
    implicit none
    
    real, allocatable :: CX(:), CY(:), CZ(:), mu(:)
    real:: mu_tot
    
    call read_CM("Apophis_CM.in", CX, CY, CZ)
    
    print *, "Data read from file:"
    print *, "X coordinates:", CX
    print *, "Y coordinates:", CY
    print *, "Z coordinates:", CZ
    
    call read_mu("Apophis_mu.in", mu)
    print *, "Mu values read from file:"
    print *, mu
    !
    mu_tot = sum(mu)
    print *, "Total mu value:", mu_tot
    ! Clean up allocated arrays
    deallocate(CX, CY, CZ, mu)
    print *, "Finished reading files."


end program main