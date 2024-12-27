<?php
include("./includes/headed.php");
include("./includes/function.php");
if(isset($_POST['login']))
{
    include("./includes/db_conn.php");
    $username = mysqli_real_escape_string($conn, $_POST['username']);
    $password = mysqli_real_escape_string($conn, $_POST['password']);
    
    $query = mysqli_query($conn, "SELECT * FROM loginn WHERE username='$username' AND password='$password'");
    $no = mysqli_num_rows($query);
    
    if($no > 0)
    {
        echo "Login successful";
        header('Location: combinedpage.php');
        exit();
    }
    else
    {
       my_alert();
    }
}
?>

<div class="main" >
    <div class="navbar">
        <div class="icon">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Armed_forces_logo.svg/2048px-Armed_forces_logo.svg.png" alt="Military Logo" class="logo-img" style="width: 80px; height:80px;">
            <h2 class="logo" style="font-size: 75px;">Military Forces</h2>
            <ul class="menu1" style=" margin-left: 235px;     width: 185px;">
                <li><a href="home.html">HOME</a></li>
                <li><a href="about.html">ABOUT</a></li>
            </ul>
        </div>
        <div class="btn">
            <!-- <button class="cn"><a href="#">JOIN US</a></button> -->
        </div>
    </div> 
    <div class="content">
        <h1>Exploring the world of <br><span>Military forces</span> <br></h1>
        <p class="par">  <br> 
            <div class="menu2">
                <ul>
                    <li>Military Soldiers</li>
                    <li>Military Units</li>
                    <li>Military Equipments</li>
                    <li>Operations</li>
                    <li>Communication between units</li>
                </ul>
            </div>
            <br></p>

        <form name="form" method="POST">
            <div class="form">
                <h2>Login Here</h2>
                <input type="text" name="username" placeholder="Enter Username Here" required>
                <input type="password" name="password" placeholder="Enter Password Here" required>
                <button type="submit" name="login" class="btnn">Login</button>
            </div>
        </form>
    </div>
</div>
<?php
include("./includes/footer.php");
?>
