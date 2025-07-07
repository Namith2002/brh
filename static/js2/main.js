// JavaScript for Breast Cancer Detection Web App

document.addEventListener('DOMContentLoaded', function() {
    // Animate prediction bars on results page
    const predictionBars = document.querySelectorAll('.bar-fill');
    if (predictionBars.length > 0) {
        setTimeout(() => {
            predictionBars.forEach(bar => {
                const targetWidth = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = targetWidth;
                }, 100);
            });
        }, 300);
    }
    
    // File input display
    const fileInput = document.getElementById('imageFile');
    const fileNameDisplay = document.getElementById('file-name-display');
    
    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                fileNameDisplay.textContent = this.files[0].name;
                
                // Preview image if possible
                const reader = new FileReader();
                reader.onload = function(e) {
                    const previewImg = document.createElement('img');
                    previewImg.src = e.target.result;
                    previewImg.className = 'preview-image';
                    
                    // Remove any existing preview
                    const existingPreview = document.querySelector('.preview-image');
                    if (existingPreview) {
                        existingPreview.remove();
                    }
                    
                    // Add new preview before the submit button
                    const submitBtn = document.querySelector('.submit-btn');
                    if (submitBtn) {
                        submitBtn.parentNode.insertBefore(previewImg, submitBtn);
                    }
                };
                reader.readAsDataURL(this.files[0]);
            } else {
                fileNameDisplay.textContent = 'No file selected';
                const existingPreview = document.querySelector('.preview-image');
                if (existingPreview) {
                    existingPreview.remove();
                }
            }
        });
    }
});

document.addEventListener('DOMContentLoaded', function() {
    // Init
    $('.image-section').hide();
    $('.image-exp').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('.image-exp').hide();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    
});
