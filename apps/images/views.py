from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse, reverse_lazy
from django.views.generic import View, CreateView, DetailView, ListView, UpdateView

from .models import ImageFile, ImageSet

class ImageSetCreateView(LoginRequiredMixin, CreateView):
    model = ImageSet
    fields = ['name', 'description', 'public']
    
    def form_valid(self, form):
        if not ImageSet.objects.filter(name=form.instance.name).exists():
            form.instance.user = self.request.user
            form.instance.dirpath = form.instance.get_dirpath()
            return super().form_valid(form)
        else:
            form.add_error(
                'name',
                f"Imageset with name {form.cleaned_data['name']} already exists in dataset. \
                    Add more images to that imageset, if required."
            )
            return HttpResponseRedirect(reverse('images:imageset_create_url'))

class ImageSetDetailView(LoginRequiredMixin, DetailView):
    model = ImageSet
    context_object_name = 'imageset'
    
class ImageSetListView(LoginRequiredMixin, ListView):
    model = ImageSet
    context_object_name = 'imagesets'
    paginate_by: int = 10

    def get_queryset(self):
        return super().get_queryset().filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user_imagesets = ImageSet.objects.filter(
            user=self.request.user).order_by('-created')
        context["user_imagesets"] = user_imagesets
        return context
    
class ImageSetUpdateView(LoginRequiredMixin, UpdateView):
    model = ImageSet
    fields = ['name', 'description', 'public']

    def form_valid(self, form):
        if not ImageSet.objects.filter(name=form.instance.name).exists():
            return super().form_valid(form)
        print("entered in else")
        form.add_error(
            'name',
            f"Imageset with name {form.cleaned_data['name']} already exists in dataset. \
                     Add more images to that imageset, if required."
        )
        context = {
            'form': form
        }
        return render(self.request, 'images/imageset_form.html', context)

    def get_success_url(self):
        return reverse('images:imageset_detail_url', kwargs={'pk': self.object.id})
    
class ImagesUploadView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        imageset_id = self.kwargs.get("pk")
        imageset = get_object_or_404(ImageSet, id=imageset_id)
        context = {
            'imageset': imageset,
        }
        return render(request, 'images/imagefile_form.html', context)

    def post(self, request, *args, **kwargs):
        imageset_id = self.kwargs.get("pk")
        imageset = get_object_or_404(ImageSet, id=imageset_id)
        if self.request.method == 'POST':
            images = [
                self.request.FILES.get("file[%d]" % i)
                for i in range(len(self.request.FILES))
            ]
            for img in images:
                if not ImageFile.objects.filter(name=img.name, image_set=imageset).exists():
                    ImageFile.objects.create(
                        name=img.name, image=img, image_set=imageset)

                else:
                    print(f"Image {img.name} already exists in the imageset.")

            message = f"Uploading images to the Imageset: {imageset}. \
                Automatic redirect to the images list after completion."

            redirect_to = reverse_lazy(
                "images:images_list_url", args=[imageset.id])
            return JsonResponse({"result": "result",
                                "message": message,
                                 "redirect_to": redirect_to,
                                 "files_length": len(images),
                                 },
                                status=200,
                                content_type="application/json"
                                )